import chess
import chess.pgn
import chess.engine
import zstandard
import json
from io import TextIOWrapper
from multiprocessing import Pool, cpu_count

PGN_FILE = "lichess_db_standard_rated_2025-08.pgn.zst"
ENGINE_PATH = "/usr/bin/stockfish"
OUTPUT_FILE = "chess_data.jsonl"

NUM_TOP_MOVES = 5
SEARCH_DEPTH = 12
NUM_PROCESSES = min(8, cpu_count())


def decompress_pgn(zst_file):
    """Yield chess.pgn.Game objects from .zst file."""
    with open(zst_file, "rb") as fh:
        dctx = zstandard.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            text_stream = TextIOWrapper(reader, encoding="utf-8")
            game = chess.pgn.read_game(text_stream)
            while game:
                yield game
                game = chess.pgn.read_game(text_stream)


def analyze_move(args):
    """Analyze a single move on a board."""
    fen, move_uci = args
    engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
    board = chess.Board(fen)

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


def extract_game_data_parallel(game):
    board = game.board()
    moves = list(game.mainline_moves())
    args_list = []

    for move in moves:
        fen = board.fen()
        args_list.append((fen, move.uci()))
        board.push(move)

    with Pool(NUM_PROCESSES) as pool:
        return pool.map(analyze_move, args_list)


def main():
    with open(OUTPUT_FILE, "w") as out:
        for game in decompress_pgn(PGN_FILE):
            game_data = extract_game_data_parallel(game)
            for entry in game_data:
                out.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    main()
