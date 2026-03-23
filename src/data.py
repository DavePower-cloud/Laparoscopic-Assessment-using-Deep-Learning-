from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def validate_dataset_path(dataset_path: Path) -> None:
    """
    Ensure the dataset path exists and contains class folders.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    if not any(d.is_dir() for d in dataset_path.iterdir()):
        raise ValueError(f"No class folders found in dataset path: {dataset_path}")


def load_video(
    video_path: Path,
    img_size: Tuple[int, int] = (128, 128),
    num_frames: int = 60,
) -> np.ndarray:
    """
    Load a video, resize frames, normalise pixel values,
    and pad with blank frames if shorter than num_frames.
    """
    frames: List[np.ndarray] = []
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    try:
        while len(frames) < num_frames:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, img_size)
            frame = frame.astype("float32") / 255.0
            frames.append(frame)

    finally:
        cap.release()

    while len(frames) < num_frames:
        frames.append(np.zeros((img_size[1], img_size[0], 3), dtype="float32"))

    return np.array(frames, dtype="float32")


def load_dataset(
    dataset_path: Path,
    img_size: Tuple[int, int] = (128, 128),
    num_frames: int = 60,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load a class-folder video dataset.

    Returns
    -------
    X : np.ndarray
        Video tensor of shape (N, T, H, W, C)
    y : np.ndarray
        Integer class labels
    class_names : list[str]
        Sorted class names
    """
    validate_dataset_path(dataset_path)

    class_names = sorted([d.name for d in dataset_path.iterdir() if d.is_dir()])

    X = []
    y = []

    for label_idx, class_name in enumerate(class_names):
        class_dir = dataset_path / class_name

        for video_file in sorted(class_dir.iterdir()):
            if video_file.is_file() and video_file.suffix.lower() in VIDEO_EXTENSIONS:
                try:
                    video_array = load_video(
                        video_file,
                        img_size=img_size,
                        num_frames=num_frames,
                    )
                    X.append(video_array)
                    y.append(label_idx)
                except Exception as exc:
                    print(f"Skipping {video_file.name}: {exc}")

    if not X:
        raise ValueError(f"No valid video files loaded from dataset path: {dataset_path}")

    return np.array(X, dtype="float32"), np.array(y, dtype="int64"), class_names
