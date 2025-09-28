#!/usr/bin/env python3
"""
split_pgns.py

Usage:
    python3 split_pgns.py --input big.pgn --out_dir small_pgns --n_files 90 --min_elo 1600

What it does:
- Streams games from an input PGN file (no full-file decompressing).
- Keeps only games where both WhiteElo and BlackElo >= min_elo.
- Writes games (as complete PGN text) into n_files output files in round-robin order.
- Each output file is a valid PGN file (no game is split).
"""

import os
import argparse
import chess.pgn


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", required=True, help="Input PGN file (extracted).")
    p.add_argument(
        "--out_dir",
        "-o",
        default="split_pgns",
        help="Directory to save small PGN files.",
    )
    p.add_argument(
        "--n_files",
        "-n",
        type=int,
        default=90,
        help="Number of output PGN files to create.",
    )
    p.add_argument(
        "--min_elo",
        type=int,
        default=1600,
        help="Minimum Elo (both players) to include a game.",
    )
    return p.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def iter_games(pgn_path):
    """Yield chess.pgn.Game objects from a PGN file (streaming)."""
    with open(pgn_path, "r", encoding="utf-8", errors="ignore") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            yield game


def write_split(input_pgn, out_dir, n_files, min_elo):
    ensure_dir(out_dir)
    # Open n_files file handles
    outs = []
    for i in range(n_files):
        path = os.path.join(out_dir, f"part_{i:03d}.pgn")
        outs.append(open(path, "w", encoding="utf-8"))

    written = 0
    kept = 0
    rr_index = 0

    for idx, game in enumerate(iter_games(input_pgn)):
        written += 1
        # Validate both elos
        try:
            we = int(game.headers.get("WhiteElo", "0"))
            be = int(game.headers.get("BlackElo", "0"))
        except Exception:
            # skip games with invalid/missing Elo headers
            continue

        if we < min_elo or be < min_elo:
            continue

        # Also skip incomplete games (no valid result or no moves)
        result = game.headers.get("Result", "")
        if result not in ("1-0", "0-1", "1/2-1/2"):
            continue
        if not list(game.mainline_moves()):
            continue

        # Write full PGN text into the next output file (round-robin)
        outs[rr_index].write(str(game) + "\n\n")
        rr_index = (rr_index + 1) % n_files
        kept += 1

        if kept % 10000 == 0:
            print(f"[split] processed {written} games, kept {kept} ...")

    for fh in outs:
        fh.close()

    print(
        f"[split] done. total scanned: {written}, total kept: {kept}. files: {n_files}"
    )


def main():
    args = parse_args()
    write_split(args.input, args.out_dir, args.n_files, args.min_elo)


if __name__ == "__main__":
    main()
