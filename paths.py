import os
from pathlib import Path


def _read_envrc_value(name):
    envrc = Path(__file__).resolve().parent / ".envrc"
    if not envrc.exists():
        return None

    for line in envrc.read_text().splitlines():
        line = line.strip()
        if line.startswith(f"export {name}="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")

    return None


def load_path(name, default=None):
    value = os.environ.get(name) or _read_envrc_value(name)
    if value:
        return Path(value).expanduser()

    if default is not None:
        return Path(default).expanduser()

    raise RuntimeError(f"{name} is not set. Run `direnv allow` or set it in .envrc.")


def urban_anomalies_path(default="/data/Datasets/UrbanAnomalies/"):
    return load_path("URBAN_ANOMALIES_PATH", os.environ.get("DATAPATH") or default)


def geolife_path(default="/data/Datasets/Geolife 1.3/"):
    return load_path("GEOLIFE_PATH", default)


def ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
