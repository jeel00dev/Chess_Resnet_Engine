#!/usr/bin/env python3
"""
selfplay_generate_fens.py

Usage example:
  python selfplay_generate_fens.py \
    --checkpoint runs/exp1/ckpt_epoch3.pt \
    --vocab vocab.json \
    --out-dir selfplay_games \
    --num-games 50 \
    --mcts-sims 80 \
    --channels 128 --blocks 8 --batch-size 1

Outputs (for each game i):
  selfplay_games/game_0001.fen      # newline FENs (initial pos + after each ply)
  selfplay_games/game_0001.meta.json

Notes:
- The model architecture (channels & blocks) must match the checkpoint's architecture.
- If checkpoint has full dict saved under 'model_state_dict', this script handles it.
"""

import os
import json
import time
import argparse
import math
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import chess


# ----------------------------
# Model (same small ResNet as training)
# ----------------------------
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + x
        out = self.relu(out)
        return out


class ChessResNet(nn.Module):
    def __init__(
        self, in_channels: int, channels: int, num_blocks: int, action_size: int
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(
            in_channels, channels, kernel_size=3, padding=1, bias=False
        )
        self.bn_in = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.blocks = nn.ModuleList([ResBlock(channels) for _ in range(num_blocks)])

        # policy head
        self.policy_conv = nn.Conv2d(channels, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, action_size)

        # value head
        self.value_conv = nn.Conv2d(channels, 8, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(8)
        self.value_fc1 = nn.Linear(8 * 8 * 8, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x):
        out = self.conv_in(x)
        out = self.bn_in(out)
        out = self.relu(out)
        for b in self.blocks:
            out = b(out)
        p = self.policy_conv(out)
        p = self.policy_bn(p)
        p = self.relu(p)
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)
        v = self.value_conv(out)
        v = self.value_bn(v)
        v = self.relu(v)
        v = v.view(v.size(0), -1)
        v = self.value_fc1(v)
        v = F.relu(v)
        v = self.value_fc2(v)
        v = torch.tanh(v).squeeze(-1)
        return p, v


# ----------------------------
# Board encoder (FEN -> 18x8x8 numpy)
# ----------------------------
def encode_board(fen: str):
    b = chess.Board(fen)
    planes = np.zeros((18, 8, 8), dtype=np.float32)
    for sq, piece in b.piece_map().items():
        rank = chess.square_rank(sq)
        file = chess.square_file(sq)
        row = 7 - rank
        col = file
        kind = piece.piece_type
        color = piece.color
        plane_idx = (0 if color else 6) + (kind - 1)
        planes[plane_idx, row, col] = 1.0
    planes[12, :, :] = 1.0 if b.has_kingside_castling_rights(chess.WHITE) else 0.0
    planes[13, :, :] = 1.0 if b.has_queenside_castling_rights(chess.WHITE) else 0.0
    planes[14, :, :] = 1.0 if b.has_kingside_castling_rights(chess.BLACK) else 0.0
    planes[15, :, :] = 1.0 if b.has_queenside_castling_rights(chess.BLACK) else 0.0
    planes[16, :, :] = 1.0 if b.turn == chess.WHITE else 0.0
    if b.ep_square is not None:
        ep_file = chess.square_file(b.ep_square)
        planes[17, :, :] = 0.0
        planes[17, :, ep_file] = 1.0
    else:
        planes[17, :, :] = 0.0
    return planes


# ----------------------------
# Single-process MCTS (uses model for priors & value)
# ----------------------------
class MCTSNode:
    def __init__(self, prior=0.0):
        self.prior = float(prior)
        self.N = 0
        self.W = 0.0
        self.children = {}

    def Q(self):
        return 0.0 if self.N == 0 else (self.W / self.N)


class MCTS:
    def __init__(self, net, vocab, device, cpuct=1.0):
        self.net = net
        self.vocab = vocab  # move->index
        self.device = device
        self.cpuct = float(cpuct)

    def _get_priors_and_value(self, board: chess.Board):
        planes = encode_board(board.fen())
        x = torch.from_numpy(planes).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, value = self.net(x)
            logits = logits.squeeze(0).cpu().numpy()
            logits = logits - logits.max()
            probs = np.exp(logits)
            probs = probs / (probs.sum() + 1e-12)
            value = float(value.item())
        priors = {}
        total = 0.0
        for mv in board.legal_moves:
            u = mv.uci()
            if u in self.vocab:
                p = probs[self.vocab[u]]
            else:
                p = 1e-8
            priors[u] = p
            total += p
        if total > 0:
            for k in priors:
                priors[k] /= total
        else:
            for k in priors:
                priors[k] = 1.0 / len(priors)
        return priors, value

    def run(
        self, root_board: chess.Board, num_sims=80, dirichlet_alpha=0.03, epsilon=0.25
    ):
        root = MCTSNode()
        priors, _ = self._get_priors_and_value(root_board)
        for mv, p in priors.items():
            root.children[mv] = MCTSNode(prior=p)
        legal_moves = list(root.children.keys())
        if len(legal_moves) > 0:
            noise = np.random.dirichlet([dirichlet_alpha] * len(legal_moves))
            for i, mv in enumerate(legal_moves):
                root.children[mv].prior = (
                    root.children[mv].prior * (1 - epsilon) + noise[i] * epsilon
                )

        for _ in range(num_sims):
            node = root
            board = root_board.copy()
            path = []
            # selection
            while len(node.children) > 0:
                best_score = -float("inf")
                best_mv = None
                best_child = None
                sqrt_sum = math.sqrt(sum(c.N for c in node.children.values()) + 1e-12)
                for mv, child in node.children.items():
                    U = self.cpuct * child.prior * sqrt_sum / (1 + child.N)
                    score = child.Q() + U
                    if score > best_score:
                        best_score = score
                        best_mv = mv
                        best_child = child
                if best_mv is None:
                    break
                board.push(chess.Move.from_uci(best_mv))
                path.append((node, best_mv))
                node = best_child
                if node.N == 0:
                    break
            # expansion/eval
            if board.is_game_over():
                res = board.result()
                if res == "1-0":
                    value = 1.0
                elif res == "0-1":
                    value = -1.0
                else:
                    value = 0.0
            else:
                priors, value = self._get_priors_and_value(board)
                for mv, p in priors.items():
                    if mv not in node.children:
                        node.children[mv] = MCTSNode(prior=p)
            # backprop
            for parent, mv in reversed(path):
                child = parent.children[mv]
                child.N += 1
                child.W += value
                value = -value
        visits = {mv: child.N for mv, child in root.children.items()}
        total_visits = sum(visits.values())
        if total_visits > 0:
            policy = {mv: n / total_visits for mv, n in visits.items()}
        else:
            policy = {mv: 1.0 / len(visits) for mv in visits}
        return policy


# ----------------------------
# Self-play loop (model vs itself) and saving FEN files
# ----------------------------
def play_game_and_save(
    model, vocab, device, out_dir, index, mcts_sims=80, max_halfmoves=512, greedy=True
):
    """Play one game (model vs model). Save FENs and meta file."""
    model.eval()
    board = chess.Board()
    mcts = MCTS(model, vocab, device)
    start_time = time.time()
    fens = []
    fens.append(board.fen())  # initial
    ply = 0
    while not board.is_game_over() and ply < max_halfmoves:
        policy_probs = mcts.run(board, num_sims=mcts_sims)
        # choose move
        if greedy:
            # pick move with max visit probability
            best_mv = max(policy_probs.items(), key=lambda kv: kv[1])[0]
        else:
            # sample according to probabilities
            moves = list(policy_probs.keys())
            probs = np.array([policy_probs[m] for m in moves], dtype=np.float64)
            probs = probs / (probs.sum() + 1e-12)
            choice = np.random.choice(len(moves), p=probs)
            best_mv = moves[choice]
        board.push(chess.Move.from_uci(best_mv))
        fens.append(board.fen())
        ply += 1
    duration = time.time() - start_time
    # result
    result = board.result() if board.is_game_over() else "*"
    # write files
    game_fname = os.path.join(out_dir, f"game_{index:04d}.fen")
    meta_fname = os.path.join(out_dir, f"game_{index:04d}.meta.json")
    with open(game_fname, "w", encoding="utf-8") as fh:
        for fen in fens:
            fh.write(fen + "\\n")
    meta = {
        "result": result,
        "plies": ply,
        "duration_s": duration,
        "checkpoint": getattr(model, "_loaded_from", None),
    }
    with open(meta_fname, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return game_fname, meta_fname, meta


# ----------------------------
# Load checkpoint helper
# ----------------------------
def load_model_from_checkpoint(checkpoint_path, device, channels, blocks, action_size):
    # build model
    model = ChessResNet(
        in_channels=18, channels=channels, num_blocks=blocks, action_size=action_size
    )
    model.to(device)
    sd = torch.load(checkpoint_path, map_location=device)
    # handle both saved dict and raw state_dict
    if isinstance(sd, dict) and "model_state_dict" in sd:
        state = sd["model_state_dict"]
    else:
        # assume sd is a state_dict
        state = sd
    model.load_state_dict(state)
    # attach origin info for metadata
    model._loaded_from = checkpoint_path
    return model


# ----------------------------
# Main CLI
# ----------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt)")
    p.add_argument("--vocab", required=True, help="vocab.json (move->index)")
    p.add_argument("--out-dir", required=True, help="Directory to write game FEN files")
    p.add_argument("--num-games", type=int, default=10)
    p.add_argument("--mcts-sims", type=int, default=80)
    p.add_argument(
        "--greedy",
        action="store_true",
        help="Use greedy selection (argmax) instead of sampling",
    )
    p.add_argument(
        "--channels",
        type=int,
        default=128,
        help="Model channels (must match checkpoint)",
    )
    p.add_argument(
        "--blocks",
        type=int,
        default=8,
        help="Model residual blocks (must match checkpoint)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="(unused) placeholder for compatibility",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.vocab, "r", encoding="utf-8") as fh:
        vocab = json.load(fh)

    action_size = len(vocab)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = load_model_from_checkpoint(
        args.checkpoint,
        device,
        channels=args.channels,
        blocks=args.blocks,
        action_size=action_size,
    )
    model.eval()

    print(f"Loaded model from {args.checkpoint}. Action size={action_size}")

    for i in range(1, args.num_games + 1):
        print(f"Playing game {i}/{args.num_games} ...")
        game_file, meta_file, meta = play_game_and_save(
            model,
            vocab,
            device,
            args.out_dir,
            i,
            mcts_sims=args.mcts_sims,
            greedy=args.greedy,
        )
        print("Saved:", game_file, meta_file, "result:", meta["result"])


if __name__ == "__main__":
    main()
