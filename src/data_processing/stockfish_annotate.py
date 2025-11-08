#!/usr/bin/env python3
"""
Stockfish Annotate to JSONL: Annotate PGN games with Stockfish analysis (top-N moves, policy, eval).

USAGE:
    python stockfish_annotate_json.py <pgn_file> <stockfish_path> [options]

ARGUMENTS:
    <pgn_file>: Path to input PGN file.
    <stockfish_path>: Path to Stockfish binary.

OPTIONS:
    --time <seconds>   Time (seconds) per analysis (default 0.05).
    --depth <int>      Alternative to --time: search to fixed depth (overrides time if provided).
    --top_n <int>      Number of top moves to extract (default 3).
    --threads <int>    Stockfish Threads option (optional).
    --hash <MB>        Stockfish Hash (MB) option (optional).
    --output <file>    Output JSONL file (default: <pgn_file>.jsonl).
Example run

python stockfish_annotate_simple.py games.pgn /usr/bin/stockfish \
--time 0.05 --top_n 3 --output games.jsonl
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
    """Convert a chess.engine.Score (centipawn or mate) to a 0..1 'win probability'.
    - mate -> 1.0 (if positive) or 0.0 (if negative)
    - centipawn -> logistic mapping 1/(1+10^(-cp/400))
    """
    if score is None:
        return 0.5
    # If score is a PovScore (typical), convert to relative (from side to move -> White perspective)
    try:
        # score may be a PovScore or a cp/mate object
        if score.is_mate():
            return 1.0 if score.mate() > 0 else 0.0
        cp = score.score(mate_score=100000)  # returns int centipawns
        if cp is None:
            return 0.5
        # logistic-style mapping (similar to Elo -> win prob)
        return 1 / (1 + 10 ** (-cp / 400))
    except Exception:
        # fallback
        return 0.5


def extract_multipv_entries(info):
    """
    Normalize different engine.analyse return formats into a list of info dicts,
    each having 'score' and 'pv' keys (when available).
    - python-chess may return a dict with 'multipv' key, or a list of dicts.
    """
    if info is None:
        return []
    if isinstance(info, list):
        return info
    if isinstance(info, dict):
        # Some engines/embed may place 'multipv' in info as a list
        if "multipv" in info and isinstance(info["multipv"], list):
            return info["multipv"]
        # If only single entry returned as dict, return it as single-element list
        return [info]
    # unknown format
    return []


def main():
    parser = argparse.ArgumentParser(description="Stockfish PGN to JSONL annotator")
    parser.add_argument("pgn_file", type=Path, help="Input PGN file")
    parser.add_argument("stockfish_path", type=Path, help="Stockfish binary path")
    parser.add_argument(
        "--time", type=float, default=0.05, help="Time per analysis (s)"
    )
    parser.add_argument(
        "--depth", type=int, default=None, help="Fixed search depth (overrides time)"
    )
    parser.add_argument("--top_n", type=int, default=3, help="Top moves to extract")
    parser.add_argument(
        "--threads", type=int, default=None, help="Stockfish Threads option"
    )
    parser.add_argument(
        "--hash", type=int, default=None, help="Stockfish Hash (MB) option"
    )
    parser.add_argument("--output", type=Path, default=None, help="Output JSONL")
    args = parser.parse_args()

    if not args.pgn_file.exists():
        sys.exit(f"Error: {args.pgn_file} not found")
    if not args.stockfish_path.exists():
        sys.exit(f"Error: Stockfish at {args.stockfish_path} not found")

    out_file = args.output or args.pgn_file.with_suffix(".jsonl")

    engine = None
    total_positions = 0
    try:
        engine = chess.engine.SimpleEngine.popen_uci(str(args.stockfish_path))

        # Optional engine configuration
        config = {}
        if args.threads is not None:
            config["Threads"] = args.threads
        if args.hash is not None:
            config["Hash"] = args.hash
        if config:
            try:
                engine.configure(config)
            except Exception:
                # Not all engines accept all options; ignore non-fatal configure errors
                pass

        with (
            open(args.pgn_file, encoding="utf-8") as pgn_f,
            open(out_file, "w", encoding="utf-8") as out_f,
        ):
            game = chess.pgn.read_game(pgn_f)
            pbar = tqdm(desc="Games", unit="game")
            while game:
                headers = game.headers
                result = headers.get("Result", "*")
                try:
                    white_elo = int(headers.get("WhiteElo", 0)) or 0
                except Exception:
                    white_elo = 0
                try:
                    black_elo = int(headers.get("BlackElo", 0)) or 0
                except Exception:
                    black_elo = 0

                board = game.board()
                pbar.set_postfix({"positions": total_positions})
                for move in game.mainline_moves():
                    board.push(move)
                    fen = board.fen()
                    # Analyse the current position with Stockfish
                    limit = (
                        chess.engine.Limit(time=args.time)
                        if args.depth is None
                        else chess.engine.Limit(depth=args.depth)
                    )
                    # Ask for multipv
                    try:
                        info = engine.analyse(board, limit, multipv=args.top_n)
                    except chess.engine.EngineTerminatedError:
                        # engine died unexpectedly
                        raise
                    except Exception:
                        # If analyse fails, record a fallback entry with minimal info
                        info = None

                    entries = extract_multipv_entries(info)

                    # Build top_moves & policy
                    policy = {}
                    top_moves = []
                    # entries may be a list of info dicts (multipv order) or a single dict
                    for entry in entries:
                        # some entries contain 'pv' (list of Move objects), 'score' etc.
                        pv = entry.get("pv")
                        if pv and len(pv) > 0:
                            uci_move = pv[0].uci()
                        else:
                            # If pv missing, try 'pv' key as str or use 'move' key
                            uci_move = None
                            # try possible keys
                            if "pv" in entry and isinstance(entry["pv"], str):
                                uci_move = entry["pv"]
                            elif "move" in entry:
                                try:
                                    uci_move = entry["move"].uci()
                                except Exception:
                                    uci_move = str(entry["move"])
                        entry_score = entry.get("score")
                        # score might be PovScore; get relative
                        try:
                            if hasattr(entry_score, "relative"):
                                entry_score = entry_score.relative
                        except Exception:
                            pass
                        prob = score_to_prob(entry_score)
                        if uci_move:
                            policy[uci_move] = round(prob, 4)
                            top_moves.append(
                                {"move": uci_move, "score": round(prob, 4)}
                            )

                    # Determine played move eval:
                    played_uci = move.uci()
                    played_eval = None
                    # If played move present in policy, use that
                    if played_uci in policy:
                        played_eval = policy[played_uci]
                    else:
                        # fallback: use best-entry score if available
                        if len(entries) > 0:
                            best_score = entries[0].get("score")
                            try:
                                if hasattr(best_score, "relative"):
                                    best_score = best_score.relative
                            except Exception:
                                pass
                            played_eval = round(score_to_prob(best_score), 4)
                        else:
                            played_eval = 0.5

                    record = {
                        "fen": fen,
                        "played_move": played_uci,
                        "played_eval": played_eval,
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

    finally:
        if engine is not None:
            try:
                engine.quit()
            except Exception:
                pass

    print(f"Done: {total_positions} positions written to {out_file}")


if __name__ == "__main__":
    main()
