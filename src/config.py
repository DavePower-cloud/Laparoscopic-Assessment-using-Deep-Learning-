from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class AppConfig:
    train_dataset_path: Path
    test_dataset_path: Path
    output_dir: Path
    img_size: Tuple[int, int]
    num_frames: int
    num_folds: int
    epochs: int
    batch_size: int
    num_classes: int
    seed: int


def get_default_config() -> AppConfig:
    return AppConfig(
        train_dataset_path=Path("data/train"),
        test_dataset_path=Path("data/test"),
        output_dir=Path("outputs"),
        img_size=(128, 128),
        num_frames=60,
        num_folds=5,
        epochs=5,
        batch_size=1,
        num_classes=3,
        seed=42,
    )
