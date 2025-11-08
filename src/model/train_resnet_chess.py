#!/usr/bin/env python3
import torch
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models, optimizers, losses

# ============================================
# Load tensor.pt and convert for TF
# ============================================
data = torch.load("data/processed/train_tensor/tensor.pt")

boards = data["boards"].float()  # (N,12,8,8)
boards = boards.permute(0, 2, 3, 1).numpy()  # → (N,8,8,12) TF format

move_index = data["move_index"].long().numpy()  # (N,)
eval_targets = data["eval"].float().numpy()  # (N,1)


# ============================================
# Residual Block
# ============================================
def ResidualBlock(x, filters):
    skip = x
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Add()([skip, x])
    x = layers.ReLU()(x)
    return x


# ============================================
# ResNet Model (TensorFlow / Keras)
# ============================================
def build_model(filters=128, blocks=6):
    inp = layers.Input(shape=(8, 8, 12))  # Channels-last

    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    for _ in range(blocks):
        x = ResidualBlock(x, filters)

    # Policy Head
    p = layers.Conv2D(32, 1, use_bias=False)(x)
    p = layers.BatchNormalization()(p)
    p = layers.ReLU()(p)
    p = layers.Flatten()(p)
    policy_logits = layers.Dense(4096, activation=None, name="policy_logits")(p)

    # Value Head
    v = layers.Conv2D(16, 1, use_bias=False)(x)
    v = layers.BatchNormalization()(v)
    v = layers.ReLU()(v)
    v = layers.Flatten()(v)
    v = layers.Dense(64, activation="relu")(v)
    value = layers.Dense(1, activation="sigmoid", name="value")(v)

    model = models.Model(inputs=inp, outputs=[policy_logits, value])
    return model


model = build_model()
model.summary()

# ============================================
# Loss + Optimizer
# ============================================
model.compile(
    optimizer=optimizers.Adam(1e-3),
    loss={
        "policy_logits": losses.SparseCategoricalCrossentropy(from_logits=True),
        "value": losses.MeanSquaredError(),
    },
    loss_weights={"policy_logits": 1.0, "value": 1.0},
    metrics={"policy_logits": "accuracy", "value": "mse"},
)

# ============================================
# Train
# ============================================
model.fit(
    boards,
    {"policy_logits": move_index, "value": eval_targets},
    epochs=20,
    batch_size=128,
    validation_split=0.05,
)

# ============================================
# Export to TensorFlow SavedModel (.pb)
# ============================================
tf.saved_model.save(model, "exported_model")

# ============================================
# Compress to .pb.gz
# ============================================
import shutil, gzip

with open("exported_model/saved_model.pb", "rb") as f_in:
    with gzip.open("exported_model.pb.gz", "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

print("✅ Exported model to exported_model.pb.gz")
