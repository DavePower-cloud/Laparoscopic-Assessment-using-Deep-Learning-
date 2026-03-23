from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: Path) -> None:
    """
    Create a directory if it does not already exist.
    """
    path.mkdir(parents=True, exist_ok=True)


def save_json(payload: Dict[str, Any], output_path: Path) -> None:
    """
    Save a Python dictionary as JSON.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def validate_num_classes(num_classes: int) -> None:
    """
    Validate class count.
    """
    if num_classes not in {2, 3}:
        raise ValueError("num_classes must be 2 or 3.")


def pretty_print_config(config: Any) -> None:
    """
    Print configuration in a readable format.
    """
    print("Configuration")
    print("-------------")
    for key, value in vars(config).items():
        print(f"{key}: {value}")
