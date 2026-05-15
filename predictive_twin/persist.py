from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import joblib


def save_model(model: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path) -> Any:
    return joblib.load(path)


def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())
