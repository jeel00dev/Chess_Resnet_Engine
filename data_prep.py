import chess
import chess.pgn
import chess.engine
import zstandard
import json
from io import TextIOWrapper

PGN_FILE = "lichess_db_standard_rated_2025-08.pgn.zst"
ENGINE_PATH = "/usr/bin/stockfish"
OUTPUT_FILE = "chess_data.jsonl"

# Config
NUM_TOP_MOVES = 5
SEARCH_DEPTH = 15


def decompress_pgn(zst_file):
    """Stream PGN lines from a .zst file."""
    with open(zst_file, "rb") as fh:
        dctx = zstandard.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            text_stream = TextIOWrapper(reader, encoding="utf-8")
            game = chess.pgn.read_game(text_stream)
            while game:
                yield game
                game = chess.pgn.read_game(text_stream)


def extract_game_data(game, engine):
    board = game.board()
    moves = list(game.mainline_moves())
    game_data = []

    for move in moves:
        fen = board.fen()
        played_move = move.uci()

        info = engine.analyse(
            board, chess.engine.Limit(depth=SEARCH_DEPTH), multipv=NUM_TOP_MOVES
        )

        top_moves = []
        policy = {}
        total_score = 0

        for entry in info:
            mv = entry["pv"][0]
            score = entry["score"].pov(board.turn).score(mate_score=1000) / 100.0
            top_moves.append({"move": mv.uci(), "score": round(score, 4)})
            total_score += max(score, 0)

        if total_score > 0:
            for m in top_moves:
                policy[m["move"]] = round(m["score"] / total_score, 4)
        else:
            for m in top_moves:
                policy[m["move"]] = round(1.0 / len(top_moves), 4)

        board.push(move)
        played_eval_info = engine.analyse(board, chess.engine.Limit(depth=SEARCH_DEPTH))
        played_eval = (
            played_eval_info["score"].pov(board.turn).score(mate_score=1000) / 100.0
        )
        played_eval = round(played_eval, 4)
        board.pop()

        game_data.append(
            {
                "fen": fen,
                "played_move": played_move,
                "played_eval": played_eval,
                "top_moves": top_moves,
                "policy": policy,
            }
        )

        board.push(move)

    return game_data


def main():
    engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
    with open(OUTPUT_FILE, "w") as out:
        for game in decompress_pgn(PGN_FILE):
            data = extract_game_data(game, engine)
            for entry in data:
                out.write(json.dumps(entry) + "\n")
    engine.quit()


if __name__ == "__main__":
    main()
