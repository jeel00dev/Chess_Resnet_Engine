#!/usr/bin/env python3
import torch
import numpy as np
import tensorflow as tf
import chess

# -----------------------------
# Load dataset (.pt)
# -----------------------------
data = torch.load("data/processed/train_tensor/tensor.pt")

boards = data["boards"].float()  # (N,12,8,8)
move_idx = data["move_index"].long()  # (N,)
eval_true = data["eval"].float()  # (N,1)

# Convert boards to TensorFlow format (N,8,8,12)
boards_tf = boards.permute(0, 2, 3, 1).numpy()

# -----------------------------
# Load the saved TF model
# -----------------------------
model = tf.saved_model.load("exported_model")  # folder containing saved_model.pb
infer = model.signatures["serving_default"]


# -----------------------------
# Helper: decode move_index → UCI
# -----------------------------
def index_to_uci(idx):
    from_sq = idx // 64
    to_sq = idx % 64
    ff, fr = from_sq % 8, from_sq // 8
    tf, tr = to_sq % 8, to_sq // 8
    return f"{chr(ff + 97)}{fr + 1}{chr(tf + 97)}{tr + 1}"


# -----------------------------
# Test a few random positions
# -----------------------------
import random

samples = random.sample(range(len(boards_tf)), 5)

for i in samples:
    x = boards_tf[i : i + 1]  # keep batch dim
    gt_move = index_to_uci(move_idx[i].item())
    gt_eval = eval_true[i].item()

    # Model forward pass
    out = infer(tf.constant(x))
    policy_logits = out["policy_logits"].numpy()[0]  # (4096,)
    value_pred = out["value"].numpy()[0][0]  # scalar

    # Predicted move index
    pred_idx = int(np.argmax(policy_logits))
    pred_move = index_to_uci(pred_idx)

    print("--------------------------------------------------")
    print(f"Position index    : {i}")
    print(f"Ground Truth Move : {gt_move}")
    print(f"Model Move        : {pred_move}")
    print(f"Ground Truth Eval : {round(gt_eval, 4)}")
    print(f"Model Eval        : {round(value_pred, 4)}")

    if pred_move == gt_move:
        print("✅ MOVE MATCHES (model is reading board data correctly)")
    else:
        print("⚠️ MOVE DOES NOT MATCH (model is not perfect / still learning)")

    print()
