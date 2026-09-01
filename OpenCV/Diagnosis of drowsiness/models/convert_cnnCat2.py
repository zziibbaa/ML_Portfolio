import h5py
import numpy as np

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Dropout,
    Flatten,
    Dense
)


# ============================================================
# 1. Paths
# ============================================================

h5_path = "cnnCat2.h5"
output_path = "cnnCat2.keras"


# ============================================================
# 2. Build the original architecture
# ============================================================

model = Sequential([
    Input(shape=(24, 24, 1)),

    Conv2D(
        32,
        kernel_size=(3, 3),
        activation="relu",
        name="conv2d_1"
    ),

    MaxPooling2D(
        pool_size=(1, 1),
        name="max_pooling2d_1"
    ),

    Conv2D(
        32,
        kernel_size=(3, 3),
        activation="relu",
        name="conv2d_2"
    ),

    MaxPooling2D(
        pool_size=(1, 1),
        name="max_pooling2d_2"
    ),

    Conv2D(
        64,
        kernel_size=(3, 3),
        activation="relu",
        name="conv2d_3"
    ),

    MaxPooling2D(
        pool_size=(1, 1),
        name="max_pooling2d_3"
    ),

    Dropout(
        0.25,
        name="dropout_1"
    ),

    Flatten(
        name="flatten_1"
    ),

    Dense(
        128,
        activation="relu",
        name="dense_1"
    ),

    Dropout(
        0.5,
        name="dropout_2"
    ),

    Dense(
        2,
        activation="softmax",
        name="dense_2"
    )
])


# ============================================================
# 3. Read and transfer weights
# ============================================================

with h5py.File(h5_path, "r") as f:

    weights_group = f["model_weights"]

    def get_weights(layer_name):

        group = weights_group[layer_name][layer_name]

        kernel = np.array(
            group["kernel:0"]
        )

        bias = np.array(
            group["bias:0"]
        )

        return [kernel, bias]


    # ------------------------------------------
    # Transfer Conv2D weights
    # ------------------------------------------

    model.get_layer("conv2d_1").set_weights(
        get_weights("conv2d_1")
    )

    model.get_layer("conv2d_2").set_weights(
        get_weights("conv2d_2")
    )

    model.get_layer("conv2d_3").set_weights(
        get_weights("conv2d_3")
    )


    # ------------------------------------------
    # Transfer Dense weights
    # ------------------------------------------

    model.get_layer("dense_1").set_weights(
        get_weights("dense_1")
    )

    model.get_layer("dense_2").set_weights(
        get_weights("dense_2")
    )


# ============================================================
# 4. Compile
# ============================================================

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# 5. Save converted model
# ============================================================

model.save(output_path)


print()
print("========================================")
print("Model conversion successful!")
print("========================================")
print(f"Input shape : {model.input_shape}")
print(f"Output shape: {model.output_shape}")
print(f"Saved to    : {output_path}")
print("========================================")