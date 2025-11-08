import json
import random
import torch
import chess

# ==== CONFIG ====
TENSOR_FILE = "data/processed/train_tensor/tensor.pt"
JSONL_FILE = "data/processed/stockfish_annotated/annotated.json"
NUM_CHECKS = 100
# ================


def move_index_to_uci(move_index):
    """Convert 0..4095 index → UCI string like 'e2e4'."""
    from_sq = move_index // 64
    to_sq = move_index % 64
    return chess.square_name(from_sq) + chess.square_name(to_sq)


def main():
    print("Loading tensor.pt ...")
    data = torch.load(TENSOR_FILE)

    boards = data["boards"]  # (N,12,8,8)
    move_index = data["move_index"]  # (N,)
    eval_targets = data["eval"]  # (N,1)

    print("Loading JSONL...")
    fens = []
    played_moves = []
    evals = []

    with open(JSONL_FILE, "r") as f:
        for line in f:
            obj = json.loads(line)
            fens.append(obj["fen"])
            played_moves.append(obj["played_move"])
            evals.append(float(obj["played_eval"]))

    N = min(len(fens), len(move_index))
    print(f"Dataset size aligned to: {N} positions\n")

    indices = random.sample(range(N), min(NUM_CHECKS, N))
    mismatches = 0

    for i in indices:
        uci_from_tensor = move_index_to_uci(int(move_index[i].item()))
        uci_from_json = played_moves[i]

        eval_tensor = float(eval_targets[i].item())
        eval_json = evals[i]

        move_ok = uci_from_tensor == uci_from_json
        eval_ok = abs(eval_tensor - eval_json) < 1e-3  # tolerance

        if not (move_ok and eval_ok):
            mismatches += 1
            print(f"[Mismatch @ index {i}]")
            print(f"  FEN: {fens[i]}")
            print(f"  JSON move:   {uci_from_json}")
            print(f"  Tensor move: {uci_from_tensor}")
            print(f"  JSON eval:   {eval_json}")
            print(f"  Tensor eval: {eval_tensor}\n")

    print("=====================================")
    print(f"Checked {len(indices)} random positions.")
    print(f"Matches: {len(indices) - mismatches}")
    print(f"Mismatches: {mismatches}")
    print("=====================================")


if __name__ == "__main__":
    main()
