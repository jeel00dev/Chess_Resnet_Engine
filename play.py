import chess
import chess.engine
import chess.pgn
import time

ENGINE_PATH = "/home/jeel00dev/Projects/chess_engine/lc0/build/release/lc0"
WEIGHTS_PATH = "/home/jeel00dev/Downloads/maia-1600.pb.gz"
PGN_FILE = "selfplay_maia.pgn"

# Start engine
engine = chess.engine.SimpleEngine.popen_uci([ENGINE_PATH, "-w", WEIGHTS_PATH])
print("Engine started successfully!\n")

num_games = 3  # Try a smaller number first
move_time = 0.5  # seconds per move (you can increase this later)

with open(PGN_FILE, "w") as pgn_out:
    for g in range(num_games):
        print(f"Starting Game {g + 1}/{num_games}")
        board = chess.Board()
        game = chess.pgn.Game()
        node = game

        while not board.is_game_over():
            result = engine.play(board, chess.engine.Limit(time=move_time))
            board.push(result.move)
            node = node.add_variation(result.move)

        game.headers["Event"] = "Maia Self-Play"
        game.headers["Result"] = board.result()
        print(f"Game {g + 1} finished: {board.result()}")
        print(game, file=pgn_out)

engine.quit()
print(f"\nAll games saved to: {PGN_FILE}")
