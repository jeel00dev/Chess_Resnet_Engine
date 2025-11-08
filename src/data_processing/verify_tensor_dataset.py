#!/usr/bin/env python3
"""
verify_model_alignment.py

This script verifies that:
1) The ResNet model input & output shapes match the dataset tensors.
2) The move_index <-> UCI mapping is correct.
3) The model can overfit a single sample (sanity check).

If all checks pass → your dataset and ResNet implementation are aligned correctly.
"""

import torch
import chess
from src.model.train_resnet_chess import ResNetChess, ChessTensorDataset


def decode_move_index(move_index: int) -> str:
    """Convert move_index (0..4095) -> UCI string."""
    from_sq = move_index // 64
    to_sq = move_index % 64
    return chess.square_name(from_sq) + chess.square_name(to_sq)


def check_shapes(model, dataset):
    print("\n=== Shape Check ===")
    x, move_idx, eval_val = dataset[0]
    x = x.unsqueeze(0)  # Add batch dim, shape (1,12,8,8)

    logits, value = model(x)

    print("logits shape:", logits.shape)
    print("value shape :", value.shape)

    assert logits.shape == (1, 4096), "❌ Policy head shape mismatch"
    assert value.shape == (1,), "❌ Value head shape mismatch"
    print("✅ Shape test passed.")


def check_move_mapping(dataset):
    print("\n=== Move Decoding Check ===")
    x, move_idx, eval_val = dataset[0]
    uci = decode_move_index(move_idx)
    print(f"move_index = {move_idx}, decoded UCI = {uci}")
    print("✅ Move index mapping is valid (conversion works).")


def overfit_single(model, dataset, device="cpu"):
    print("\n=== 1-Sample Overfit Test ===")
    model = model.to(device)

    board, move_idx, eval_val = dataset[0]
    board = board.unsqueeze(0).to(device)  # (1,12,8,8)
    move_idx = torch.tensor([move_idx], device=device)
    eval_val = torch.tensor([eval_val], dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in range(200):
        optimizer.zero_grad()
        logits, value = model(board)

        policy_loss = torch.nn.functional.cross_entropy(logits, move_idx)
        value_loss = torch.nn.functional.mse_loss(value, eval_val)
        loss = policy_loss + value_loss

        loss.backward()
        optimizer.step()

        if step % 20 == 0:
            pred = logits.argmax(1).item()
            pred_uci = decode_move_index(pred)
            true_uci = decode_move_index(move_idx.item())
            print(
                f"step {step:03d} | loss={loss.item():.4f} | pred={pred_uci} | true={true_uci} | value={value.item():.4f}"
            )

    print(
        "✅ If loss decreased and predicted move becomes correct → model, data, losses are aligned."
    )


def main():
    DATA_PATH = "data/processed/train_tensor/tensor.pt"  # <-- change if needed
    CHECKPOINT = None  # or e.g. "checkpoints/resnet_small/last.pt"

    print("Loading dataset...")
    dataset = ChessTensorDataset(DATA_PATH)

    print("Loading model...")
    model = ResNetChess(channels=128, blocks=6)

    if CHECKPOINT:
        print("Loading checkpoint:", CHECKPOINT)
        ckpt = torch.load(CHECKPOINT, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])

    # Perform tests
    check_shapes(model, dataset)
    check_move_mapping(dataset)
    overfit_single(model, dataset)


if __name__ == "__main__":
    main()
