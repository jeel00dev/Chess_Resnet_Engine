#!/usr/bin/env python3
"""
jsonl_to_lmdb.py

Usage examples:
1) Build vocabulary from input jsonl(s) and save to vocab.json:
   python jsonl_to_lmdb.py --build-vocab --input games/*.jsonl --vocab-out vocab.json

2) Convert jsonl -> LMDB using existing vocab:
   python jsonl_to_lmdb.py --input games/*.jsonl --vocab vocab.json --out dataset.lmdb

3) Do both in one go:
   python jsonl_to_lmdb.py --build-vocab --input games/*.jsonl --vocab-out vocab.json --out dataset.lmdb

Dependencies:
pip install python-chess lmdb numpy torch tqdm
"""

import argparse
import json
import os
import glob
from collections import Counter, defaultdict
import lmdb
import pickle
from tqdm import tqdm
import numpy as np
import chess


# ---------------------------
# Encoding: board -> planes
# ---------------------------
def encode_board(fen):
    """
    Encode a FEN into a tensor of shape (18, 8, 8) float32
    Planes:
      0-5:  white P,N,B,R,Q,K
      6-11: black p,n,b,r,q,k
      12: white can castle kingside
      13: white can castle queenside
      14: black can castle kingside
      15: black can castle queenside
      16: side to move (all ones if white to move, else zeros)
      17: en-passant file plane (1 on a file where ep capture is possible, else zeros)
    Returns: numpy array (18,8,8), dtype float32
    """
    b = chess.Board(fen)
    planes = np.zeros((18, 8, 8), dtype=np.float32)
    piece_map = b.piece_map()
    for sq, piece in piece_map.items():
        rank = chess.square_rank(sq)
        file = chess.square_file(sq)
        # board indexing: row 0 is rank 8 -> we flip rank
        row = 7 - rank
        col = file
        kind = piece.piece_type  # 1..6
        color = piece.color  # True=white
        plane_idx = (0 if color else 6) + (kind - 1)
        planes[plane_idx, row, col] = 1.0
    # castling rights
    planes[12, :, :] = 1.0 if b.has_kingside_castling_rights(chess.WHITE) else 0.0
    planes[13, :, :] = 1.0 if b.has_queenside_castling_rights(chess.WHITE) else 0.0
    planes[14, :, :] = 1.0 if b.has_kingside_castling_rights(chess.BLACK) else 0.0
    planes[15, :, :] = 1.0 if b.has_queenside_castling_rights(chess.BLACK) else 0.0
    # side to move
    planes[16, :, :] = 1.0 if b.turn == chess.WHITE else 0.0
    # en-passant file
    if b.ep_square is not None:
        ep_file = chess.square_file(b.ep_square)
        planes[17, :, :] = 0.0
        planes[17, :, ep_file] = 1.0
    else:
        planes[17, :, :] = 0.0
    return planes


# ---------------------------
# Vocab building
# ---------------------------
def build_vocab_from_jsonl(paths, max_moves=None):
    """
    Scan input JSONL files and collect distinct move strings from:
      - 'policy' keys (if present)
      - 'top_moves' entries
      - 'played_move'
    Return: dict move->index
    """
    counter = Counter()
    count = 0
    for p in paths:
        with open(p, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                obj = json.loads(line)
                # policy dict
                if "policy" in obj and isinstance(obj["policy"], dict):
                    for mv in obj["policy"].keys():
                        counter[mv] += 1
                # top_moves
                if "top_moves" in obj and isinstance(obj["top_moves"], list):
                    for entry in obj["top_moves"]:
                        if "move" in entry:
                            counter[entry["move"]] += 1
                # played_move
                if "played_move" in obj and obj["played_move"]:
                    counter[obj["played_move"]] += 1
                count += 1
                if max_moves and count >= max_moves:
                    break
    # sort by frequency and create index (reserve 0 for unknown if needed)
    moves_sorted = [m for m, _ in counter.most_common()]
    vocab = {m: i for i, m in enumerate(moves_sorted)}
    return vocab


# ---------------------------
# Convert and write LMDB
# ---------------------------
def convert_jsonl_to_lmdb(
    input_paths, lmdb_path, vocab, map_unknown_to=None, write_topk_policy=False
):
    """
    Convert JSONL -> LMDB using given vocab (move->index).
    Each LMDB record stored (pickled) as a dict:
      {
        'board': np.float32 (18,8,8),
        'policy': np.float32 (action_space,)   # if policy present AND action_space <= 10000
        'played_index': int (index of played_move)   # always stored if played_move available
        'value': float (game outcome from white perspective: 1, 0, -1)
        'meta': { 'fen':..., 'white_elo':..., 'black_elo':... }
      }
    If policy vector would be very large (>10000), script currently stores sparse list of (indices, probs) instead.
    """
    action_space = len(vocab)
    # choose whether to store dense vector (faster) or sparse
    store_dense = action_space <= 10000
    os.makedirs(os.path.dirname(os.path.abspath(lmdb_path)), exist_ok=True)
    # estimate map size conservatively (grow if necessary)
    map_size = (
        1 << 36
    )  # 64 GiB max allowed; LMDB map_size must be >= data size; pick large to avoid errors.
    env = lmdb.open(
        lmdb_path, map_size=map_size, subdir=False, readonly=False, meminit=False
    )
    txn = env.begin(write=True)
    key_idx = 0
    committed = 0
    write_batch = 1000
    try:
        for p in input_paths:
            with open(p, "r", encoding="utf-8") as fh:
                for line in tqdm(fh, desc=f"Converting {p}"):
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    fen = obj.get("fen")
                    try:
                        board_planes = encode_board(fen)
                    except Exception as e:
                        # skip malformed fen
                        print("Skipping malformed FEN:", fen, "err:", e)
                        continue
                    record = {}
                    record["board"] = board_planes  # numpy float32
                    # value target from 'game_result' (string "1-0", "0-1", "1/2-1/2")
                    gr = obj.get("game_result")
                    if gr is None:
                        value = 0.0
                    else:
                        if gr.strip() in ("1-0", "1.0-0.0"):
                            value = 1.0
                        elif gr.strip() in ("0-1", "0.0-1.0"):
                            value = -1.0
                        else:
                            # draws
                            value = 0.0
                    record["value"] = float(value)
                    # meta
                    record["meta"] = {
                        "fen": fen,
                        "white_elo": obj.get("white_elo"),
                        "black_elo": obj.get("black_elo"),
                    }
                    # played_move index
                    played = obj.get("played_move")
                    if played:
                        if played in vocab:
                            record["played_index"] = int(vocab[played])
                        else:
                            if map_unknown_to is not None:
                                record["played_index"] = int(map_unknown_to)
                            else:
                                record["played_index"] = -1
                    else:
                        record["played_index"] = -1
                    # policy (if available)
                    if "policy" in obj and isinstance(obj["policy"], dict):
                        pol = obj["policy"]
                        if store_dense:
                            vec = np.zeros((action_space,), dtype=np.float32)
                            for mv, prob in pol.items():
                                if mv in vocab:
                                    vec[vocab[mv]] = float(prob)
                                else:
                                    # ignore unknown move or optionally collect into leftover index
                                    pass
                            # normalize in case the input wasn't normalized (safety)
                            s = vec.sum()
                            if s > 0:
                                vec /= s
                            record["policy"] = vec
                        else:
                            # sparse storage
                            idxs = []
                            vals = []
                            for mv, prob in pol.items():
                                if mv in vocab:
                                    idxs.append(vocab[mv])
                                    vals.append(float(prob))
                            idxs = np.array(idxs, dtype=np.int32)
                            vals = np.array(vals, dtype=np.float32)
                            # normalize sparse probs
                            s = vals.sum()
                            if s > 0:
                                vals /= s
                            record["policy_sparse_idx"] = idxs
                            record["policy_sparse_val"] = vals
                    else:
                        # no policy provided: optional fallback — create one-hot at played_index
                        if record["played_index"] != -1:
                            if store_dense:
                                vec = np.zeros((action_space,), dtype=np.float32)
                                vec[record["played_index"]] = 1.0
                                record["policy"] = vec
                            else:
                                record["policy_sparse_idx"] = np.array(
                                    [record["played_index"]], dtype=np.int32
                                )
                                record["policy_sparse_val"] = np.array(
                                    [1.0], dtype=np.float32
                                )
                    # serialize and write
                    key = f"{key_idx:08d}".encode("ascii")
                    val = pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL)
                    txn.put(key, val)
                    key_idx += 1
                    if key_idx % write_batch == 0:
                        txn.commit()
                        committed += write_batch
                        txn = env.begin(write=True)
        # final commit
        txn.commit()
        print(f"Done. Wrote {key_idx} records to {lmdb_path}")
    finally:
        env.close()


# ---------------------------
# LMDB Dataset wrapper for PyTorch
# ---------------------------
class LMDBChessDataset:
    """
    Simple LMDB dataset to read the stored pickled records.
    Returns: (board_tensor, policy_tensor_or_sparse, value, meta)
    board_tensor: torch.float32 tensor (C,8,8)
    policy_tensor: numpy float32 dense vector if present, else None (or sparse idx/val arrays)
    """

    def __init__(self, lmdb_path, transform=None):
        import torch

        self.torch = torch
        self.env = lmdb.open(
            lmdb_path, readonly=True, lock=False, readahead=False, meminit=False
        )
        with self.env.begin() as txn:
            self.length = 0
            cursor = txn.cursor()
            for _ in cursor:
                self.length += 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        import torch

        key = f"{idx:08d}".encode("ascii")
        with self.env.begin() as txn:
            v = txn.get(key)
            if v is None:
                raise IndexError(idx)
            record = pickle.loads(v)
        board = record["board"]  # numpy
        # convert to torch
        board_t = self.torch.from_numpy(board).to(dtype=self.torch.float32)
        policy = record.get("policy", None)
        if policy is not None:
            policy_t = self.torch.from_numpy(policy).to(dtype=self.torch.float32)
        else:
            # return sparse if available
            idxs = record.get("policy_sparse_idx", None)
            vals = record.get("policy_sparse_val", None)
            if idxs is not None:
                policy_t = (idxs, vals)
            else:
                policy_t = None
        value = float(record.get("value", 0.0))
        played_index = int(record.get("played_index", -1))
        meta = record.get("meta", {})
        return board_t, policy_t, float(value), int(played_index), meta


# ---------------------------
# CLI
# ---------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Convert chess JSONL -> LMDB with board encoding."
    )
    ap.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Input JSONL files or glob (quote glob on shell)",
    )
    ap.add_argument("--out", default="dataset.lmdb", help="Output LMDB file")
    ap.add_argument(
        "--build-vocab",
        action="store_true",
        help="Scan files and build a move vocabulary first",
    )
    ap.add_argument(
        "--vocab-out", default="vocab.json", help="If building vocab, write here"
    )
    ap.add_argument(
        "--vocab",
        default=None,
        help="If provided, use this vocab json instead of building",
    )
    ap.add_argument(
        "--max-sample-scan",
        type=int,
        default=None,
        help="During vocab build scan only this many lines (debug)",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    # expand globs
    input_paths = []
    for p in args.input:
        if any(ch in p for ch in ["*", "?", "["]):
            input_paths.extend(sorted(glob.glob(p)))
        else:
            input_paths.append(p)
    if len(input_paths) == 0:
        raise SystemExit("No input files matched.")
    if args.build_vocab:
        print("Building vocab from input...")
        vocab = build_vocab_from_jsonl(input_paths, max_moves=args.max_sample_scan)
        with open(args.vocab_out, "w", encoding="utf-8") as fh:
            json.dump(vocab, fh, indent=2)
        print(f"Saved vocab (size={len(vocab)}) to {args.vocab_out}")
    if args.vocab:
        with open(args.vocab, "r", encoding="utf-8") as fh:
            vocab = json.load(fh)
        print(f"Loaded vocab size {len(vocab)}")
    elif args.build_vocab:
        vocab = (
            {k: v for k, v in json.load(open(args.vocab_out)).items()}
            if os.path.exists(args.vocab_out)
            else None
        )
        if vocab is None:
            # already created `vocab` above; keep it
            pass
    else:
        raise SystemExit("Provide --vocab or --build-vocab to create one.")
    # convert
    convert_jsonl_to_lmdb(input_paths, args.out, vocab)
    print("Conversion complete. Example usage snippet for training:")
    print(
        """
    # Example reading dataset:
    from jsonl_to_lmdb import LMDBChessDataset
    ds = LMDBChessDataset('dataset.lmdb')
    import torch
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    for board, policy, value, played_idx, meta in loader:
        # board shape: (B, C, 8, 8)
        # policy: either dense tensor (B, action_dim) or tuple(sparse)
        pass
    """
    )


if __name__ == "__main__":
    main()
