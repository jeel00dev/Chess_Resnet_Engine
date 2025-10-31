#!/usr/bin/env python3
"""
Subsample PGN: Extract 1M high-ELO games for quality data.
Usage: python subsample_pgn.py <input_pgn> --output <out_pgn> --min_elo 2000 --max_games 1000000
"""

import argparse
import chess.pgn
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="High-ELO PGN subsampler")
    parser.add_argument("input_pgn", type=Path, help="Input PGN file")
    parser.add_argument("--output", type=Path, required=True, help="Output PGN")
    parser.add_argument("--min_elo", type=int, default=2000, help="Min avg ELO")
    parser.add_argument("--max_games", type=int, default=1000000, help="Max games")
    args = parser.parse_args()

    if not args.input_pgn.exists():
        raise FileNotFoundError(f"{args.input_pgn} not found")

    # Balance outcomes (rough: 40% white wins, 40% black, 20% draws)
    outcome_counts = defaultdict(int)
    max_per_outcome = args.max_games // 2  # Adjust for balance

    count = 0
    with (
        open(args.input_pgn, "r", encoding="utf-8") as f,
        open(args.output, "w", encoding="utf-8") as out,
    ):
        game = chess.pgn.read_game(f)
        pbar = tqdm(desc="Subsampling", unit="game")
        while game and count < args.max_games:
            headers = game.headers
            white_elo = int(headers.get("WhiteElo", 0) or 0)
            black_elo = int(headers.get("BlackElo", 0) or 0)
            avg_elo = (white_elo + black_elo) / 2
            result = headers.get("Result", "*")

            if avg_elo >= args.min_elo and outcome_counts[result] < max_per_outcome:
                out.write(str(game) + "\n\n")
                outcome_counts[result] += 1
                count += 1

            pbar.update(1)
            pbar.set_postfix({"total": count, "avg_elo": f"{avg_elo:.0f}"})
            game = chess.pgn.read_game(f)

    print(f"Saved {count} high-ELO games to {args.output}")
    print(f"Outcome balance: {dict(outcome_counts)}")


if __name__ == "__main__":
    main()
