from pathlib import Path
import yaml


def load_config(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_output_dir(cfg):
    out = Path(cfg.get("output_dir", "outputs/dg_pocformer"))
    out.mkdir(parents=True, exist_ok=True)
    return out
