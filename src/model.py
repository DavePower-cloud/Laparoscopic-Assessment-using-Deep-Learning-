from __future__ import annotations

from typing import Tuple

from tensorflow.keras.layers import (
    BatchNormalization,
    Conv3D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling3D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def build_3dcnn_model(
    input_shape: Tuple[int, int, int, int],
    num_classes: int,
):
    """
    Build and compile a 3DCNN model.
    """
    model = Sequential()

    model.add(
        Conv3D(
            64,
            (3, 3, 3),
            activation="relu",
            padding="same",
            input_shape=input_shape,
        )
    )
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv3D(128, (3, 3, 3), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv3D(256, (3, 3, 3), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv3D(512, (3, 3, 3), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation="relu"))

    if num_classes == 2:
        model.add(Dense(1, activation="sigmoid"))
        loss = "binary_crossentropy"
    else:
        model.add(Dense(num_classes, activation="softmax"))
        loss = "categorical_crossentropy"

    model.compile(
        optimizer=Adam(),
        loss=loss,
        metrics=["accuracy"],
    )

    return model
