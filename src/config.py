import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

_ACTS_PATH = Path("config/acts.yaml")


def load_act_config(key: str, path: Path = _ACTS_PATH) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if key not in data:
        raise KeyError(f"no act config named {key!r} in {path}")
    return data[key]


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"{name} is not set. Copy .env.example to .env and fill it in."
        )
    return value
