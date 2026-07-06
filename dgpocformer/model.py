import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root mean square normalization."""
    def __init__(self, d: int, eps: float = 1e-8):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x / (x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt())) * self.w


def _build_shifted_mask(grid_size: int, win_size: int, shift: int) -> torch.Tensor:
    H = W = grid_size
    m = torch.zeros(1, H, W, 1)
    slices = (slice(0, -win_size), slice(-win_size, -shift), slice(-shift, None))
    cnt = 0
    for h in slices:
        for w in slices:
            m[:, h, w, :] = cnt
            cnt += 1
    mask = (
        m.reshape(1, H // win_size, win_size, W // win_size, win_size, 1)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(-1, win_size * win_size)
    )
    a = mask.unsqueeze(1) - mask.unsqueeze(2)
    return a.masked_fill(a != 0, -100.0).masked_fill(a == 0, 0.0)


class FactoredPE2D(nn.Module):
    """Factored 2D positional encoding using row and column embeddings."""
    def __init__(self, H: int, W: int, d: int):
        super().__init__()
        half = d // 2
        self.row_embed = nn.Parameter(torch.zeros(H, half))
        self.col_embed = nn.Parameter(torch.zeros(W, half))
        nn.init.trunc_normal_(self.row_embed, std=0.02)
        nn.init.trunc_normal_(self.col_embed, std=0.02)
        self.H, self.W = H, W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.row_embed.unsqueeze(1).expand(-1, self.W, -1).reshape(-1, self.row_embed.size(1))
        c = self.col_embed.unsqueeze(0).expand(self.H, -1, -1).reshape(-1, self.col_embed.size(1))
        return x + torch.cat([r, c], -1).unsqueeze(0)


class SwiGLUFFN(nn.Module):
    def __init__(self, d: int, expand: int = 4):
        super().__init__()
        h = d * expand
        self.norm = RMSNorm(d)
        self.gate = nn.Linear(d, h)
        self.up = nn.Linear(d, h)
        self.down = nn.Linear(h, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = self.norm(x)
        return x + self.down(F.silu(self.gate(n)) * self.up(n))


class FineWindowAttn(nn.Module):
    """Local window self-attention with relative position bias and optional cyclic shift."""
    def __init__(self, d: int, nh: int, ws: int, grid_size: int = 16, shift: int = 0):
        super().__init__()
        self.nh = nh
        self.hd = d // nh
        self.ws = ws
        self.shift = shift
        self.scl = self.hd ** -0.5
        self.norm = RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d)
        self.rpb = nn.Parameter(torch.zeros((2 * ws - 1) ** 2, nh))
        nn.init.trunc_normal_(self.rpb, std=0.02)

        c = torch.stack(torch.meshgrid(torch.arange(ws), torch.arange(ws), indexing="ij"))
        cf = c.flatten(1)
        rel = cf[:, :, None] - cf[:, None, :]
        rel = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += ws - 1
        rel[:, :, 1] += ws - 1
        rel[:, :, 0] *= 2 * ws - 1
        self.register_buffer("rel_idx", rel.sum(-1))

        if shift > 0:
            self.register_buffer("attn_mask", _build_shifted_mask(grid_size, ws, shift))
        else:
            self.attn_mask = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, d = x.shape
        H = W = int(N ** 0.5)
        ws = self.ws
        nw = (H // ws) * (W // ws)
        res = x
        xn = self.norm(x)

        if self.shift > 0:
            xn = torch.roll(
                xn.reshape(B, H, W, d), shifts=(-self.shift, -self.shift), dims=(1, 2)
            ).reshape(B, N, d)

        qkv = (
            self.qkv(xn).reshape(B, H, W, 3, d)
            .reshape(B, H // ws, ws, W // ws, ws, 3, d)
            .permute(0, 1, 3, 5, 2, 4, 6)
            .reshape(B * nw, 3, ws * ws, self.nh, self.hd)
            .permute(1, 0, 3, 2, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1) * self.scl + self.rpb[self.rel_idx].permute(2, 0, 1).unsqueeze(0))

        if self.attn_mask is not None:
            attn = (
                attn.reshape(B, nw, self.nh, ws * ws, ws * ws)
                + self.attn_mask.unsqueeze(0).unsqueeze(2)
            ).reshape(B * nw, self.nh, ws * ws, ws * ws)

        attn = attn.softmax(-1)
        out = (attn @ v).transpose(1, 2).reshape(B * nw, ws * ws, d)
        out = (
            self.proj(out)
            .reshape(B, H // ws, W // ws, ws, ws, d)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(B, H, W, d)
            .reshape(B, N, d)
        )

        if self.shift > 0:
            out = torch.roll(
                out.reshape(B, H, W, d), shifts=(self.shift, self.shift), dims=(1, 2)
            ).reshape(B, N, d)

        return res + out


class CoarseGlobalAttn(nn.Module):
    """Global self-attention over coarse tokens."""
    def __init__(self, d: int, nh: int):
        super().__init__()
        self.nh = nh
        self.hd = d // nh
        self.scl = self.hd ** -0.5
        self.norm = RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, d = x.shape
        qkv = self.qkv(self.norm(x)).reshape(B, N, 3, self.nh, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1) * self.scl).softmax(-1)
        return x + self.proj((attn @ v).transpose(1, 2).reshape(B, N, d))


class CrossGranAttn(nn.Module):
    """Cross-granularity attention."""
    def __init__(self, d_q: int, d_kv: int, nh: int):
        super().__init__()
        self.nh = nh
        self.hd = d_q // nh
        self.scl = self.hd ** -0.5
        self.q_proj = nn.Linear(d_q, nh * self.hd, bias=False)
        self.k_proj = nn.Linear(d_kv, nh * self.hd, bias=False)
        self.v_proj = nn.Linear(d_kv, nh * self.hd, bias=False)
        self.out_proj = nn.Linear(nh * self.hd, d_q)
        self.norm_q = RMSNorm(d_q)
        self.norm_kv = RMSNorm(d_kv)

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor, return_attn: bool = False):
        B, Nq, _ = x_q.shape
        Nkv = x_kv.shape[1]
        q = self.q_proj(self.norm_q(x_q)).reshape(B, Nq, self.nh, self.hd).transpose(1, 2)
        k = self.k_proj(self.norm_kv(x_kv)).reshape(B, Nkv, self.nh, self.hd).transpose(1, 2)
        v = self.v_proj(self.norm_kv(x_kv)).reshape(B, Nkv, self.nh, self.hd).transpose(1, 2)
        aw = (q @ k.transpose(-2, -1) * self.scl).softmax(-1)
        out = self.out_proj((aw @ v).transpose(1, 2).reshape(B, Nq, self.nh * self.hd))
        if return_attn:
            return out, aw.mean(dim=(1, 3))
        return out


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    rand = torch.rand(shape, dtype=x.dtype, device=x.device).floor_().div_(keep)
    return x * rand


class DualGranularityBlock(nn.Module):
    def __init__(
        self,
        d_f: int,
        d_c: int,
        nh_f: int,
        nh_c: int,
        nh_x: int,
        ws: int,
        block_idx: int = 0,
        dp_rate: float = 0.0,
    ):
        super().__init__()
        shift = ws // 2 if block_idx % 2 == 1 else 0
        self.fine_sa = FineWindowAttn(d_f, nh_f, ws, grid_size=16, shift=shift)
        self.coarse_sa = CoarseGlobalAttn(d_c, nh_c)
        self.c2f = CrossGranAttn(d_f, d_c, nh_x)
        self.f2c = CrossGranAttn(d_c, d_f, nh_x)
        self.fine_ffn = SwiGLUFFN(d_f)
        self.coarse_ffn = SwiGLUFFN(d_c)
        self.alpha_c2f = nn.Parameter(torch.ones(1))
        self.alpha_f2c = nn.Parameter(torch.ones(1))
        self.dp_rate = dp_rate

    def _dp(self, delta: torch.Tensor) -> torch.Tensor:
        return drop_path(delta, self.dp_rate, self.training) if self.dp_rate > 0 else delta

    def forward(self, x_f: torch.Tensor, x_c: torch.Tensor, return_saliency: bool = False):
        x_f = x_f + self._dp(self.fine_sa(x_f) - x_f)
        x_c = x_c + self._dp(self.coarse_sa(x_c) - x_c)
        a_c2f = F.softplus(self.alpha_c2f)
        a_f2c = F.softplus(self.alpha_f2c)

        if return_saliency:
            c2f_out, saliency = self.c2f(x_f, x_c, return_attn=True)
        else:
            c2f_out = self.c2f(x_f, x_c)

        x_f = x_f + self._dp(a_c2f * c2f_out)
        x_c = x_c + self._dp(a_f2c * self.f2c(x_c, x_f))
        x_f = x_f + self._dp(self.fine_ffn(x_f) - x_f)
        x_c = x_c + self._dp(self.coarse_ffn(x_c) - x_c)

        if return_saliency:
            return x_f, x_c, saliency
        return x_f, x_c


class DGPOCFormer(nn.Module):
    def __init__(
        self,
        num_classes: int = 4,
        d_fine: int = 48,
        d_coarse: int = 64,
        depth: int = 3,
        n_heads_fine: int = 4,
        n_heads_coarse: int = 4,
        n_heads_cross: int = 4,
        window_size: int = 4,
        head_hidden: int = 256,
        head_drop: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.d_fine = d_fine
        self.d_coarse = d_coarse
        self.depth = depth

        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
        )

        self.patch_embed = nn.Conv2d(32, d_fine, 4, 4, bias=False)
        self.fine_pe = FactoredPE2D(16, 16, d_fine)
        self.coarse_proj = nn.Linear(d_fine, d_coarse)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_coarse))
        self.coarse_pe = nn.Parameter(torch.zeros(1, 17, d_coarse))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.coarse_pe, std=0.02)

        self.blocks = nn.ModuleList([
            DualGranularityBlock(
                d_fine,
                d_coarse,
                n_heads_fine,
                n_heads_coarse,
                n_heads_cross,
                window_size,
                block_idx=i,
                dp_rate=0.0,
            )
            for i in range(depth)
        ])

        self.norm_f = RMSNorm(d_fine)
        self.norm_c = RMSNorm(d_coarse)
        self.head = nn.Sequential(
            nn.Linear(d_fine + d_coarse, head_hidden),
            nn.GELU(),
            nn.Dropout(head_drop),
            nn.Linear(head_hidden, num_classes),
        )
        self.aux_head = nn.Linear(d_fine, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor):
        B = x.shape[0]
        x = self.stem(x)
        x = self.patch_embed(x)
        x_f = self.fine_pe(x.flatten(2).transpose(1, 2))
        x_c = self.coarse_proj(F.avg_pool2d(x, 4, 4).flatten(2).transpose(1, 2))
        x_c = torch.cat([self.cls_token.expand(B, -1, -1), x_c], 1) + self.coarse_pe

        saliency = None
        for i, blk in enumerate(self.blocks):
            if i == len(self.blocks) - 1:
                x_f, x_c, saliency = blk(x_f, x_c, return_saliency=True)
            else:
                x_f, x_c = blk(x_f, x_c)

        fine_pooled = self.norm_f(x_f).mean(dim=1)
        coarse_out = self.norm_c(x_c[:, 0])
        return fine_pooled, coarse_out, saliency

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        fine_pooled, coarse_out, _ = self.forward_features(x)
        fused = torch.cat([fine_pooled, coarse_out], -1)
        logits = self.head(fused)
        if return_aux:
            return logits, self.aux_head(fine_pooled)
        return logits


def build_model_from_config(cfg: dict, num_classes: int = 4) -> DGPOCFormer:
    mcfg = cfg.get("model", cfg)
    return DGPOCFormer(
        num_classes=num_classes,
        d_fine=int(mcfg.get("d_fine", 48)),
        d_coarse=int(mcfg.get("d_coarse", 64)),
        depth=int(mcfg.get("depth", 3)),
        n_heads_fine=int(mcfg.get("n_heads_fine", 4)),
        n_heads_coarse=int(mcfg.get("n_heads_coarse", 4)),
        n_heads_cross=int(mcfg.get("n_heads_cross", 4)),
        window_size=int(mcfg.get("window_size", 4)),
        head_hidden=int(mcfg.get("head_hidden", 256)),
        head_drop=float(mcfg.get("head_drop", 0.3)),
    )


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    aux = sum(p.numel() for p in model.aux_head.parameters()) if hasattr(model, "aux_head") else 0
    return {"total": total, "auxiliary": aux, "inference": total - aux}
