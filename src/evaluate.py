from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, auc, classification_report, confusion_matrix, roc_curve
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

from config import get_default_config
from data import load_dataset
from utils import ensure_dir, save_json, validate_num_classes


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate 3DCNN for laparoscopic skill assessment.")
    parser.add_argument("--num-classes", type=int, default=None, help="Number of classes: 2 or 3.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_default_config()

    num_classes = args.num_classes if args.num_classes is not None else config.num_classes
    validate_num_classes(num_classes)

    ensure_dir(config.output_dir)

    model_path = config.output_dir / "lap_skill_3dcnn_model.h5"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    X_test, y_test, class_names = load_dataset(
        config.test_dataset_path,
        img_size=config.img_size,
        num_frames=config.num_frames,
    )

    print("Test data shape:", X_test.shape)
    print("Test labels shape:", y_test.shape)
    print("Classes:", class_names)

    if num_classes == 2:
        y_test_encoded = y_test
    else:
        y_test_encoded = to_categorical(y_test, num_classes=num_classes)

    model = load_model(model_path)

    test_loss, test_acc = model.evaluate(X_test, y_test_encoded, verbose=0)
    print("Test accuracy:", test_acc)

    y_pred_probs = model.predict(X_test)

    if num_classes == 2:
        y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    else:
        y_pred = np.argmax(y_pred_probs, axis=1)

    acc = float(accuracy_score(y_test, y_pred))
    print("Accuracy score:", acc)

    cm = confusion_matrix(y_test, y_pred)
    print("\\nConfusion Matrix:")
    print(cm)

    report_text = classification_report(y_test, y_pred, target_names=class_names)
    print("\\nClassification Report:")
    print(report_text)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(config.output_dir / "confusion_matrix.png", dpi=300)
    plt.show()

    roc_auc_value = None
    if num_classes == 2:
        y_prob_vector = y_pred_probs.flatten()
        fpr, tpr, _ = roc_curve(y_test, y_prob_vector)
        roc_auc_value = float(auc(fpr, tpr))

        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc_value:.2f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(config.output_dir / "roc_curve.png", dpi=300)
        plt.show()

    sample_idx = 0
    sample_video = X_test[sample_idx:sample_idx + 1]
    sample_true = int(y_test[sample_idx])

    sample_pred_probs = model.predict(sample_video)
    if num_classes == 2:
        sample_pred = int(sample_pred_probs.flatten()[0] > 0.5)
    else:
        sample_pred = int(np.argmax(sample_pred_probs, axis=1)[0])

    print("\\nExample Prediction")
    print("True label:", class_names[sample_true])
    print("Predicted label:", class_names[sample_pred])

    metrics_payload = {
        "num_classes": num_classes,
        "test_accuracy": float(test_acc),
        "accuracy_score": acc,
        "roc_auc": roc_auc_value,
        "class_names": class_names,
        "confusion_matrix": cm.tolist(),
        "classification_report": report_text,
        "example_prediction": {
            "true_label": class_names[sample_true],
            "predicted_label": class_names[sample_pred],
        },
    }
    save_json(metrics_payload, config.output_dir / "evaluation_metrics.json")


if __name__ == "__main__":
    main()
