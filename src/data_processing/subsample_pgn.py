#!/usr/bin/env python3
"""
Simplified PGN subsampler.
- Keeps games where average ELO >= min_elo
- Handles missing/unknown ELO (treat as 0)
- Stops after max_games
- Writes output PGN

Usage:
  python subsample_pgn.py input.pgn --output filtered.pgn --min_elo 2000 --max_games 100000
"""

import argparse
import chess.pgn
from pathlib import Path
from tqdm import tqdm


def safe_elo(value):
    """Convert ELO header to integer. Returns 0 if unknown or invalid."""
    if not value or value in {"?", "-"}:
        return 0
    try:
        return int(value)
    except ValueError:
        # Try extracting numeric characters
        digits = "".join(c for c in str(value) if c.isdigit())
        return int(digits) if digits else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_pgn", type=Path, help="Input PGN file")
    parser.add_argument("--output", type=Path, required=True, help="Output PGN file")
    parser.add_argument(
        "--min_elo", type=int, default=2000, help="Minimum average ELO to keep game"
    )
    parser.add_argument(
        "--max_games",
        type=int,
        default=100000,
        help="Stop after collecting this many games",
    )
    args = parser.parse_args()

    with (
        open(args.input_pgn, "r", encoding="utf-8", errors="replace") as src,
        open(args.output, "w", encoding="utf-8") as dst,
    ):
        pbar = tqdm(desc="Processing", unit="game")
        count = 0

        game = chess.pgn.read_game(src)
        while game and count < args.max_games:
            headers = game.headers

            # Read ELO values safely
            w = safe_elo(headers.get("WhiteElo"))
            b = safe_elo(headers.get("BlackElo"))
            avg = (w + b) / 2

            # Keep only high-level games
            if avg >= args.min_elo:
                dst.write(str(game) + "\n\n")
                count += 1

            pbar.update(1)
            pbar.set_postfix({"saved": count, "avg_elo": int(avg)})
            game = chess.pgn.read_game(src)

    print(f"Saved {count} games to {args.output}")


if __name__ == "__main__":
    main()
