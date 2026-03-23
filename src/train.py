from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
from tensorflow.keras.utils import to_categorical

from config import get_default_config
from data import load_dataset
from model import build_3dcnn_model
from utils import ensure_dir, pretty_print_config, save_json, set_seed, validate_num_classes


def parse_args():
    parser = argparse.ArgumentParser(description="Train 3DCNN for laparoscopic skill assessment.")
    parser.add_argument("--num-classes", type=int, default=None, help="Number of classes: 2 or 3.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_default_config()

    num_classes = args.num_classes if args.num_classes is not None else config.num_classes
    epochs = args.epochs if args.epochs is not None else config.epochs
    batch_size = args.batch_size if args.batch_size is not None else config.batch_size

    validate_num_classes(num_classes)
    set_seed(config.seed)
    ensure_dir(config.output_dir)

    pretty_print_config(config)
    print("Runtime overrides")
    print("-----------------")
    print(f"num_classes: {num_classes}")
    print(f"epochs: {epochs}")
    print(f"batch_size: {batch_size}")

    X_train, y_train, class_names = load_dataset(
        config.train_dataset_path,
        img_size=config.img_size,
        num_frames=config.num_frames,
    )

    if len(class_names) != num_classes:
        print(
            f"Warning: dataset has {len(class_names)} class folders but num_classes={num_classes}."
        )

    print("Training data shape:", X_train.shape)
    print("Training labels shape:", y_train.shape)
    print("Classes:", class_names)

    if num_classes == 2:
        y_train_encoded = y_train
    else:
        y_train_encoded = to_categorical(y_train, num_classes=num_classes)

    kf = KFold(n_splits=config.num_folds, shuffle=True, random_state=config.seed)

    histories: List = []
    fold_accuracies: List[float] = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), start=1):
        print(f"\\n--- Fold {fold}/{config.num_folds} ---")

        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train_encoded[train_idx], y_train_encoded[val_idx]

        model = build_3dcnn_model(
            input_shape=(config.num_frames, config.img_size[1], config.img_size[0], 3),
            num_classes=num_classes,
        )

        history = model.fit(
            X_tr,
            y_tr,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
        )

        histories.append(history)

        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        fold_accuracies.append(float(val_acc))

        print(f"Fold {fold} validation accuracy: {val_acc:.4f}")

    mean_cv_acc = float(np.mean(fold_accuracies))
    std_cv_acc = float(np.std(fold_accuracies))

    print("\\nCross-validation accuracies:", fold_accuracies)
    print("Mean CV accuracy:", mean_cv_acc)
    print("Std CV accuracy:", std_cv_acc)

    plt.figure(figsize=(12, 5))
    for i, history in enumerate(histories, start=1):
        plt.plot(history.history["accuracy"], label=f"Fold {i} Train")
        plt.plot(history.history["val_accuracy"], linestyle="--", label=f"Fold {i} Val")
    plt.title("Training and Validation Accuracy Across Folds")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.output_dir / "cv_accuracy_plot.png", dpi=300)
    plt.show()

    final_model = build_3dcnn_model(
        input_shape=(config.num_frames, config.img_size[1], config.img_size[0], 3),
        num_classes=num_classes,
    )

    final_model.fit(
        X_train,
        y_train_encoded,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )

    model_path = config.output_dir / "lap_skill_3dcnn_model.h5"
    weights_path = config.output_dir / "lap_skill_3dcnn_weights.h5"

    final_model.save(model_path)
    final_model.save_weights(weights_path)

    print("Model saved to:", model_path)
    print("Weights saved to:", weights_path)

    metrics_payload = {
        "num_classes": num_classes,
        "epochs": epochs,
        "batch_size": batch_size,
        "fold_accuracies": fold_accuracies,
        "mean_cv_accuracy": mean_cv_acc,
        "std_cv_accuracy": std_cv_acc,
        "class_names": class_names,
    }
    save_json(metrics_payload, config.output_dir / "training_metrics.json")


if __name__ == "__main__":
    main()
