#!/usr/bin/env python3
"""
json_to_tensors.py

Convert JSONL (one JSON object per line) with fields like:
{
 "fen": "...",
 "played_move": "e2e4",
 "played_eval": 0.437,
 "policy": {"e2e4": 0.437}, ...
}

to a compact PyTorch dataset saved as output.pt containing:
  - boards:         (N, 12, 8, 8)   float32  (binary channels for pieces)
  - side_to_move:   (N, 1)          float32  (1.0 = white to move, 0.0 = black)
  - castling:       (N, 4)          float32  (W-K, W-Q, B-K, B-Q -> 1/0)
  - en_passant:     (N, 64)         float32  (one-hot ep square or all zeros)
  - move_from:      (N, 64)         float32  (one-hot source square of played move)
  - move_to:        (N, 64)         float32  (one-hot destination square of played move)
  - move_index:     (N,)            int64    (from*64 + to, 0..4095)
  - eval:           (N,1)          float32  (played_eval as scalar, 0..1)

Math notes (short):
- Board channels: 12 binary channels = 6 piece types × 2 colors.
  Represent as a tensor X in R^{12×8×8}. Each channel is a 8×8 binary matrix.
  This is convenient for convolutional networks (shape matches Conv2D inputs).
- One-hot encoding: a categorical variable with k categories is represented as a vector
  with a single 1 at the category index and zeros elsewhere. This is mathematically
  an indicator vector e_i ∈ {0,1}^k. For 'from' square this is k=64.
- Move index mapping: index = from_index * 64 + to_index -> unique integer in [0,4095].
  This is simple bijection from pair (from,to) ∈ {0..63}^2 to scalar index ∈ {0..4095}.
- Loss examples:
  - Policy classification: CrossEntropyLoss(logits, move_index) where logits ∈ R^{4096}.
  - Eval regression: MSELoss(pred_eval, eval_scalar).

See inline comments for more math background.
"""

import json
import argparse
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
import chess  # python-chess
import chess.svg

# ---------------------------
# Helper math / conversion
# ---------------------------

# Mapping from piece letter to channel index (0..11).
# Channel ordering: [P,N,B,R,Q,K, p,n,b,r,q,k]  -- white pieces first (0-5), black pieces next (6-11).
# This is a small deterministic mapping used only to index the 12 channels. It's not a learned "vocab".
_piece_to_channel = {
    "P": 0,
    "N": 1,
    "B": 2,
    "R": 3,
    "Q": 4,
    "K": 5,
    "p": 6,
    "n": 7,
    "b": 8,
    "r": 9,
    "q": 10,
    "k": 11,
}


def square_index(file: int, rank: int) -> int:
    """
    Convert (file,rank) -> square index 0..63 using convention:
      file: 0..7 for a..h
      rank: 0..7 for 1..8
    We'll use index = rank*8 + file so that contiguous indices walk along ranks.
    This matches python-chess.square_index (but implemented explicitly for clarity).

    Math: this is simply a positional number in base-8.
    """
    return rank * 8 + file


def uci_to_from_to(uci: str) -> Tuple[int, int]:
    """
    Convert UCI like 'e2e4' or 'a7a8q' -> (from_index, to_index).
    We ignore promotion piece (could incorporate if you want distinct target classes).
    """
    # UCI format: from_file from_rank to_file to_rank [promotion]
    uci = uci.strip()
    if len(uci) < 4:
        raise ValueError(f"Invalid UCI move: '{uci}'")
    from_sq = uci[0:2]
    to_sq = uci[2:4]

    def sq_to_idx(sq: str) -> int:
        file_c = ord(sq[0]) - ord("a")  # 0..7
        rank_c = int(sq[1]) - 1  # 0..7
        if not (0 <= file_c < 8 and 0 <= rank_c < 8):
            raise ValueError(f"Bad square in move: {sq}")
        return square_index(file_c, rank_c)

    return sq_to_idx(from_sq), sq_to_idx(to_sq)


def from_to_to_move_index(from_idx: int, to_idx: int) -> int:
    """
    Deterministic bijection (from,to) -> single class index:
      move_index = from_idx * 64 + to_idx
    Range: 0 .. 4095

    Rationale (math): this is simply interpreting the pair (a,b) as digits in base-64.
    This mapping is invertible:
      from_idx = move_index // 64
      to_idx   = move_index % 64
    """
    return from_idx * 64 + to_idx


def fen_to_tensor(fen: str) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Convert FEN -> (board_tensor, stm, castling_vec, en_passant_onehot)
    - board_tensor: np.float32 shape (12, 8, 8). Binary channels per piece.
        For a square containing a white pawn, channel 0 at that (r,c) is 1.0.
    - stm: scalar 1.0 if white to move, 0.0 if black (we return as float)
    - castling_vec: shape (4,) order: [W_K, W_Q, B_K, B_Q], each 0/1
    - en_passant: shape (64,) one-hot of en-passant target square, all zeros if '-'

    Mathematical justification:
      - Each channel is an indicator function I_{piece_type}(square).
        The board tensor is concatenation of these indicator matrices => a sparse binary tensor.
      - This representation is linear (concatenation) and preserves spatial locality (suitable for Conv2D).
    """
    board_channels = np.zeros((12, 8, 8), dtype=np.float32)
    # Parse FEN with python-chess for safety
    b = chess.Board(fen)

    # Fill piece channels
    # python-chess: squares are 0..63 with 0 = a1, 7 = h1, 8 = a2, ... 63 = h8
    for sq in chess.SQUARES:
        piece = b.piece_at(sq)
        if piece is not None:
            # compute rank/file from sq: rank 0..7 corresponds to 1..8
            rank = chess.square_rank(sq)  # 0..7
            file = chess.square_file(sq)  # 0..7
            ch = _piece_to_channel[piece.symbol()]
            # channel ch, at (rank,file) set to 1
            # Note: many Conv models expect shape (C, H, W) with H=8 ranks from top to bottom.
            # python-chess ranks: 0->rank1 (bottom). We will use rank axis such that index 0 = rank1.
            board_channels[ch, rank, file] = 1.0

    # side to move: 1.0 white, 0.0 black
    stm = 1.0 if b.turn == chess.WHITE else 0.0

    # castling rights
    cast_vec = np.zeros((4,), dtype=np.float32)
    # order [W-K, W-Q, B-K, B-Q]
    if b.has_kingside_castling_rights(chess.WHITE):
        cast_vec[0] = 1.0
    if b.has_queenside_castling_rights(chess.WHITE):
        cast_vec[1] = 1.0
    if b.has_kingside_castling_rights(chess.BLACK):
        cast_vec[2] = 1.0
    if b.has_queenside_castling_rights(chess.BLACK):
        cast_vec[3] = 1.0

    # en-passant target square: one-hot length 64 (or all zeros if none)
    ep_onehot = np.zeros((64,), dtype=np.float32)
    if b.ep_square is not None:
        ep_onehot[b.ep_square] = 1.0

    return board_channels, float(stm), cast_vec, ep_onehot


# ---------------------------
# Main conversion pipeline
# ---------------------------


def process_jsonl_to_tensors(
    input_path: Path,
    output_path: Path,
    limit: Optional[int] = None,
):
    """
    Read JSONL, convert each record to tensors, and save aggregated tensors to output_path (torch.save).

    Design choices & math:
    - Board: (12,8,8) binary tensor per position. This is sparse (few 1's), and works well with Conv nets.
    - Move target: we provide both 'move_index' (scalar class) and 'move_from'/'move_to' one-hot matrices.
      Mathematically, training a policy head can be done with:
        CrossEntropy(logits_of_length_4096, move_index)
      Or with two softmax heads:
        CrossEntropy(logits_from_64, from_index) + CrossEntropy(logits_to_64, to_index)
      The single 4096-class approach couples from+to decisions; the two-head approach factorizes them.
    - played_eval: kept as float scalar in [0,1]. Use MSE or HuberLoss for regression.
    """
    boards = []
    stms = []
    castlings = []
    eps = []
    move_from_onehots = []
    move_to_onehots = []
    move_indices = []
    evals = []

    with open(input_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            fen = obj.get("fen")
            if fen is None:
                # skip lines without fen
                continue

            # Convert fen -> channels + extras
            board_ch, stm, cast_vec, ep_onehot = fen_to_tensor(fen)

            # Parse played move UCI; the user data uses moves like "d2d4"
            played_move = obj.get("played_move", "")
            # If policy present (top moves), you might also extract it; here we only convert played_move.
            try:
                from_idx, to_idx = uci_to_from_to(played_move)
            except Exception:
                # If played_move missing or invalid, skip (or set to default)
                # We'll skip those records for safety
                # Alternative: mark move_index = -1 and filter later
                continue

            # move one-hot vectors
            mf = np.zeros((64,), dtype=np.float32)
            mt = np.zeros((64,), dtype=np.float32)
            mf[from_idx] = 1.0
            mt[to_idx] = 1.0

            mindex = from_to_to_move_index(from_idx, to_idx)

            # played_eval -> ensure numeric in [0,1], fallback 0.5 if missing
            pe = obj.get("played_eval", None)
            try:
                pe_val = float(pe) if pe is not None else 0.5
            except Exception:
                pe_val = 0.5
            # clamp to [0,1] for safety
            pe_val = float(max(0.0, min(1.0, pe_val)))

            # append
            boards.append(board_ch)  # (12,8,8)
            stms.append([stm])  # scalar kept as shape (1,)
            castlings.append(cast_vec)  # (4,)
            eps.append(ep_onehot)  # (64,)
            move_from_onehots.append(mf)  # (64,)
            move_to_onehots.append(mt)  # (64,)
            move_indices.append(mindex)  # scalar
            evals.append([pe_val])  # shape (1,)

    # Convert lists -> torch tensors
    boards_t = torch.tensor(np.stack(boards), dtype=torch.float32)  # (N,12,8,8)
    stms_t = torch.tensor(np.stack(stms), dtype=torch.float32)  # (N,1)
    cast_t = torch.tensor(np.stack(castlings), dtype=torch.float32)  # (N,4)
    ep_t = torch.tensor(np.stack(eps), dtype=torch.float32)  # (N,64)
    mf_t = torch.tensor(np.stack(move_from_onehots), dtype=torch.float32)  # (N,64)
    mt_t = torch.tensor(np.stack(move_to_onehots), dtype=torch.float32)  # (N,64)
    idx_t = torch.tensor(
        np.array(move_indices, dtype=np.int64), dtype=torch.int64
    )  # (N,)
    eval_t = torch.tensor(np.stack(evals), dtype=torch.float32)  # (N,1)

    # Save dictionary
    dataset = {
        "boards": boards_t,
        "side_to_move": stms_t,
        "castling": cast_t,
        "en_passant": ep_t,
        "move_from": mf_t,
        "move_to": mt_t,
        "move_index": idx_t,
        "eval": eval_t,
    }

    torch.save(dataset, str(output_path))
    print(f"Saved dataset with N={boards_t.shape[0]} positions to {output_path}")


# ---------------------------
# CLI
# ---------------------------


def main_cli():
    parser = argparse.ArgumentParser(
        description="Convert chess JSONL -> PyTorch tensors"
    )
    parser.add_argument("input", type=Path, help="Input JSONL file (one JSON per line)")
    parser.add_argument("output", type=Path, help="Output .pt file")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: max number of records to process",
    )
    args = parser.parse_args()
    process_jsonl_to_tensors(args.input, args.output, limit=args.limit)


if __name__ == "__main__":
    main_cli()
