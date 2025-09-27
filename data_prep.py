import os
import chess
import chess.pgn
import chess.engine
import zstandard
import json
from io import TextIOWrapper
from multiprocessing import Process, cpu_count

PGN_ZST_FILE = "lichess_db_standard_rated_2025-08.pgn.zst"
ENGINE_PATH = "/usr/bin/stockfish"
OUTPUT_DIR = "chess_output"
NUM_CHUNKS = 5
NUM_TOP_MOVES = 5
SEARCH_DEPTH = 12


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def decompress_pgn_stream(zst_file):
    """Stream PGN objects from zst file"""
    with open(zst_file, "rb") as fh:
        dctx = zstandard.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            text_stream = TextIOWrapper(reader, encoding="utf-8")
            game = chess.pgn.read_game(text_stream)
            while game:
                yield game
                game = chess.pgn.read_game(text_stream)


def analyze_move(fen, move_uci):
    """Analyze a single move with Stockfish"""
    board = chess.Board(fen)
    engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)

    info = engine.analyse(
        board, chess.engine.Limit(depth=SEARCH_DEPTH), multipv=NUM_TOP_MOVES
    )
    top_moves = []
    total_score = 0
    policy = {}

    for entry in info:
        mv = entry["pv"][0]
        score = entry["score"].pov(board.turn).score(mate_score=1000) / 100.0
        score = round(score, 4)
        top_moves.append({"move": mv.uci(), "score": score})
        total_score += max(score, 0)

    if total_score > 0:
        for m in top_moves:
            policy[m["move"]] = round(m["score"] / total_score, 4)
    else:
        for m in top_moves:
            policy[m["move"]] = round(1.0 / len(top_moves), 4)

    if move_uci not in policy:
        board.push_uci(move_uci)
        played_eval_info = engine.analyse(board, chess.engine.Limit(depth=SEARCH_DEPTH))
        played_eval = round(
            played_eval_info["score"].pov(board.turn).score(mate_score=1000) / 100.0, 4
        )
    else:
        played_eval = next(m["score"] for m in top_moves if m["move"] == move_uci)

    engine.quit()

    return {
        "fen": fen,
        "played_move": move_uci,
        "played_eval": played_eval,
        "top_moves": top_moves,
        "policy": policy,
    }


def process_chunk(chunk_index, total_chunks):
    """Process one chunk of the PGN dataset"""
    output_file = os.path.join(OUTPUT_DIR, f"chunk_{chunk_index}.jsonl")
    print(f"[CHUNK {chunk_index}] Processing -> {output_file}")
    with open(output_file, "w") as out:
        for i, game in enumerate(decompress_pgn_stream(PGN_ZST_FILE)):
            if i % total_chunks != chunk_index:
                continue

            board = game.board()
            for move in game.mainline_moves():
                fen = board.fen()
                move_uci = move.uci()
                entry = analyze_move(fen, move_uci)
                out.write(json.dumps(entry) + "\n")
                board.push(move)
    print(f"[CHUNK {chunk_index}] Done")


def main():
    ensure_output_dir()
    processes = []

    for idx in range(NUM_CHUNKS):
        p = Process(target=process_chunk, args=(idx, NUM_CHUNKS))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("All chunks processed!")


if __name__ == "__main__":
    main()
