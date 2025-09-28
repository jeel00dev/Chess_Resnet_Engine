#!/usr/bin/env python3
"""
analyze_pgns.py

Analyze many PGN files in parallel. For each input PGN file, produce one output JSONL file
in out_dir named <pgn_basename>.jsonl.

Each JSON line structure:
{
  "fen": "...",
  "played_move": "e2e4",
  "played_eval": -0.1234,
  "top_moves": [{"move":"e2e4","score":0.3125}, ...],
  "policy": {"e2e4":0.3125, ...},
  "game_result": "1-0",
  "white_elo": 2253,
  "black_elo": 2297
}
"""

import os
import argparse
import json
import math
from multiprocessing import Pool, cpu_count
import chess
import chess.pgn
import chess.engine


def parse_args():
    p = argparse.ArgumentParser(
        description="Analyze PGN files with Stockfish and produce JSONL datasets."
    )
    p.add_argument(
        "--pgn_dir", required=True, help="Directory containing PGN files to analyze."
    )
    p.add_argument(
        "--out_dir", default="parsed_data", help="Directory to save JSONL outputs."
    )
    p.add_argument(
        "--engine", default="/usr/bin/stockfish", help="Path to stockfish executable."
    )
    p.add_argument(
        "--workers",
        type=int,
        default=min(8, cpu_count()),
        help="Number of PGN files to process in parallel.",
    )
    p.add_argument("--depth", type=int, default=12, help="Stockfish search depth.")
    p.add_argument("--top", type=int, default=5, help="Number of top moves (multipv).")
    p.add_argument(
        "--temp",
        type=float,
        default=1.0,
        help="Softmax temperature for policy (higher -> smoother).",
    )
    return p.parse_args()


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def softmax(xs, temp=1.0):
    """
    xs: list of floats in pawn units (e.g. 0.32 means +0.32 pawns).
    temp: temperature (1.0 default).
    Returns list of probabilities same length as xs.
    """
    if not xs:
        return []
    scaled = [x / temp for x in xs]  # CORRECT: do not divide by 100 again
    m = max(scaled)
    ex = [math.exp(x - m) for x in scaled]
    s = sum(ex)
    return [e / s for e in ex]


def analyze_file(args_tuple):
    pgn_path, out_dir, engine_path, depth, top_n, temp = args_tuple
    out_path = os.path.join(out_dir, os.path.basename(pgn_path) + ".jsonl")
    print(f"[worker] analyzing {pgn_path} -> {out_path}")

    engine = None
    try:
        engine = chess.engine.SimpleEngine.popen_uci(engine_path)

        with (
            open(pgn_path, "r", encoding="utf-8", errors="ignore") as pf,
            open(out_path, "w", encoding="utf-8") as out,
        ):
            g_idx = 0
            while True:
                game = chess.pgn.read_game(pf)
                if game is None:
                    break
                g_idx += 1

                # Basic validation (skip if missing/invalid Elo or result or no moves)
                try:
                    we = int(game.headers.get("WhiteElo", "0"))
                    be = int(game.headers.get("BlackElo", "0"))
                except Exception:
                    continue
                if we == 0 or be == 0:
                    continue
                result = game.headers.get("Result", "")
                if result not in ("1-0", "0-1", "1/2-1/2"):
                    continue
                moves = list(game.mainline_moves())
                if not moves:
                    continue

                board = game.board()
                for move in moves:
                    fen = board.fen()
                    move_uci = move.uci()

                    # Ask engine for top_n PVs
                    try:
                        info_list = engine.analyse(
                            board, chess.engine.Limit(depth=depth), multipv=top_n
                        )
                    except Exception as e:
                        print(
                            f"[{pgn_path}] engine analyze failed at game {g_idx}: {e}"
                        )
                        board.push(move)
                        continue

                    top_moves = []
                    raw_scores = []
                    for entry in info_list:
                        mv = entry.get("pv", [None])[0]
                        if mv is None:
                            continue
                        score_obj = entry.get("score")
                        # Score might be mate or cp
                        cp = score_obj.pov(board.turn).score(mate_score=100000)
                        if cp is None:
                            # Decide sign from mate distance if available; fallback to positive
                            mate_val = score_obj.pov(board.turn).mate()
                            if mate_val is None:
                                cp = 100000
                            else:
                                cp = 100000 if mate_val > 0 else -100000
                        cp_f = cp / 100.0  # convert to pawn units
                        raw_scores.append(cp_f)
                        top_moves.append({"move": mv.uci(), "score": round(cp_f, 4)})

                    # Compute policy via softmax on raw_scores (in pawn units)
                    if raw_scores:
                        probs = softmax(raw_scores, temp=temp)
                        # round probabilities to 4 decimals
                        policy = {
                            top_moves[i]["move"]: round(probs[i], 4)
                            for i in range(len(top_moves))
                        }
                    else:
                        policy = {}

                    # played_eval: reuse if in top_moves, else evaluate position after the played move
                    played_eval = None
                    in_top = False
                    for m in top_moves:
                        if m["move"] == move_uci:
                            played_eval = m["score"]
                            in_top = True
                            break

                    if not in_top:
                        try:
                            board.push(move)
                            pe_info = engine.analyse(
                                board, chess.engine.Limit(depth=depth)
                            )
                            cp_pe = (
                                pe_info["score"]
                                .pov(board.turn)
                                .score(mate_score=100000)
                            )
                            if cp_pe is None:
                                mate_val = pe_info["score"].pov(board.turn).mate()
                                if mate_val is None:
                                    cp_pe = 100000
                                else:
                                    cp_pe = 100000 if mate_val > 0 else -100000
                            played_eval = round(cp_pe / 100.0, 4)
                            board.pop()
                        except Exception as e:
                            # try to recover board state
                            try:
                                board.pop()
                            except Exception:
                                pass
                            played_eval = None

                    entry = {
                        "fen": fen,
                        "played_move": move_uci,
                        "played_eval": played_eval,
                        "top_moves": top_moves,
                        "policy": policy,
                        "game_result": result,
                        "white_elo": we,
                        "black_elo": be,
                    }

                    out.write(json.dumps(entry) + "\n")
                    board.push(move)  # advance to next ply

                if g_idx % 1000 == 0:
                    print(f"[{pgn_path}] processed {g_idx} games")

    except Exception as e:
        print(f"[worker] fatal error processing {pgn_path}: {e}")
    finally:
        if engine:
            try:
                engine.quit()
            except Exception:
                pass

    print(f"[worker] finished {pgn_path}")
    return out_path


def main():
    args = parse_args()
    ensure_dir(args.out_dir)

    files = [
        os.path.join(args.pgn_dir, f)
        for f in os.listdir(args.pgn_dir)
        if f.lower().endswith(".pgn")
    ]
    files.sort()
    if not files:
        print("No .pgn files found in", args.pgn_dir)
        return

    print(
        f"Found {len(files)} PGN files; running up to {args.workers} workers in parallel."
    )
    pool_args = [
        (f, args.out_dir, args.engine, args.depth, args.top, args.temp) for f in files
    ]

    # Use a Pool of workers; each analyzes one file at a time
    with Pool(processes=args.workers) as pool:
        for _ in pool.imap_unordered(analyze_file, pool_args):
            pass

    print("All files processed.")


if __name__ == "__main__":
    main()
