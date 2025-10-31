#!/usr/bin/env python3
"""
LC0 Annotate to JSONL: Annotate PGN games with LC0 analysis (top-3 moves, policy, eval).

USAGE:
    python lc0_annotate_json.py <pgn_file> <lc0_path> [options]

ARGUMENTS:
    <pgn_file>: Path to input PGN file (e.g., data/processed/high_elo_subset.pgn).
    <lc0_path>: Path to LC0 binary (e.g., /home/user/tools/lc0).

OPTIONS:
    --time <seconds>: LC0 thinking time per position (default: 0.05 for fast processing).
    --output <file>: Output JSONL file (default: <pgn_file>.jsonl).
    --top_n <int>: Number of top moves to extract (default: 3).

EXAMPLE:
    python lc0_annotate_json.py data/processed/high_elo_subset.pgn /tools/lc0 --time 0.05 --output data/processed/high_elo.jsonl

EXPECTED OUTPUT FORMAT:
One JSON line per board position (after each move), e.g.:
{
  "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
  "played_move": "e7e5",
  "played_eval": 0.3125,
  "top_moves": [
    {"move": "e7e5", "score": 0.3125},
    {"move": "d7d5", "score": 0.2857},
    {"move": "c7c5", "score": 0.2234}
  ],
  "policy": {"e7e5": 0.3125, "d7d5": 0.2857, "c7c5": 0.2234},
  "game_result": "1-0",
  "white_elo": 1968,
  "black_elo": 1893
}
- fen: FEN string of position after played move.
- played_move: UCI of the move played (e.g., "e2e4").
- played_eval: Win probability (0-1) for the played move from LC0's perspective.
- top_moves: List of top N moves with UCI and normalized score (sum to <1).
- policy: Dict of UCI to score for top N moves.
- game_result: PGN result ("1-0", "0-1", "1/2-1/2", or "*").
- white_elo/black_elo: Player ratings from PGN headers (0 if missing).

NOTES:
- Requires LC0 weights configured in the engine (set via UCI "WeightsFile").
- Processes mainline only; ignores variations.
- For 1M games, expect 1-2 days at 0.05s/move; monitor with tqdm.
- Output is streamable JSONL for easy conversion to Parquet.
"""

import argparse
import json
import sys
from pathlib import Path

import chess
import chess.pgn
import chess.engine
from tqdm import tqdm


def score_to_prob(score):
    """Convert LC0 score (centipawn or mate) to win probability (0-1)."""
    if score.is_mate():
        return 1.0 if score.mate() > 0 else 0.0
    cp = score.score(mate_score=10000)
    if cp is None:
        return 0.5
    return 1 / (1 + 10 ** (-cp / 400))  # Logistic scaling


def main():
    parser = argparse.ArgumentParser(description="LC0 PGN to JSONL annotator")
    parser.add_argument("pgn_file", type=Path, help="Input PGN file")
    parser.add_argument("lc0_path", type=Path, help="LC0 binary path")
    parser.add_argument(
        "--time", type=float, default=0.05, help="Time per analysis (s)"
    )
    parser.add_argument("--output", type=Path, default=None, help="Output JSONL")
    parser.add_argument("--top_n", type=int, default=3, help="Top moves to extract")
    args = parser.parse_args()

    if not args.pgn_file.exists():
        sys.exit(f"Error: {args.pgn_file} not found")
    if not args.lc0_path.exists():
        sys.exit(f"Error: LC0 at {args.lc0_path} not found")

    out_file = args.output or args.pgn_file.with_suffix(".jsonl")
    engine = chess.engine.SimpleEngine.popen_uci(str(args.lc0_path))
    # Configure LC0 (update WeightsFile path as needed)
    # engine.configure({"WeightsFile": "/path/to/lc0.net"})

    total_positions = 0
    with open(args.pgn_file, encoding="utf-8") as pgn_f, open(out_file, "w") as out_f:
        game = chess.pgn.read_game(pgn_f)
        pbar = tqdm(desc="Games", unit="game")
        while game:
            headers = game.headers
            result = headers.get("Result", "*")
            white_elo = int(headers.get("WhiteElo", 0)) or 0
            black_elo = int(headers.get("BlackElo", 0)) or 0

            board = game.board()
            pbar.set_postfix({"positions": total_positions})
            for move in game.mainline_moves():
                board.push(move)
                fen = board.fen()

                # LC0 analysis
                info = engine.analyse(
                    board, chess.engine.Limit(time=args.time), multipv=args.top_n
                )

                played_uci = move.uci()
                played_score = info["score"].relative
                played_eval = score_to_prob(played_score)

                # Extract policy from multipv
                policy = {}
                top_moves = []
                for entry in info.get("multipv", []):
                    pv = entry["pv"]
                    if pv:
                        uci_move = pv[0].uci()
                        entry_score = entry["score"].relative
                        prob = score_to_prob(entry_score)
                        policy[uci_move] = round(prob, 4)
                        top_moves.append({"move": uci_move, "score": round(prob, 4)})

                record = {
                    "fen": fen,
                    "played_move": played_uci,
                    "played_eval": round(played_eval, 4),
                    "top_moves": top_moves,
                    "policy": policy,
                    "game_result": result,
                    "white_elo": white_elo,
                    "black_elo": black_elo,
                }
                out_f.write(json.dumps(record) + "\n")
                total_positions += 1

            pbar.update(1)
            game = chess.pgn.read_game(pgn_f)

    engine.quit()
    print(f"Done: {total_positions} positions written to {out_file}")


if __name__ == "__main__":
    main()
