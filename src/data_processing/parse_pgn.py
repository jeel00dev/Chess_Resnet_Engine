import zstandard as zstd
import chess.pgn
import io
import os
from tqdm import tqdm


def parse_pgn_zst(input_path, output_dir, max_games=10000000, min_moves=20):
    """Decompress and parse pgn.zst file, saving only games with at least `min_moves` moves."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"parsed_games_{min_moves}+moves.pgn")

    with open(input_path, "rb") as f:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")
            saved_count = 0
            processed_count = 0

            with open(output_path, "w", encoding="utf-8") as out:
                game = chess.pgn.read_game(text_stream)

                while game and saved_count < max_games:
                    processed_count += 1

                    # Count total moves in the game
                    move_count = game.end().board().fullmove_number

                    # Only write games with 20 or more moves
                    if move_count >= min_moves:
                        out.write(str(game) + "\n\n")
                        saved_count += 1

                    # Print progress every 500 processed games
                    if processed_count % 500 == 0:
                        print(
                            f"Processed {processed_count} games... saved {saved_count} valid games so far."
                        )

                    game = chess.pgn.read_game(text_stream)

    print(
        f"Finished! Saved {saved_count} games (with ≥{min_moves} moves) to {output_path}"
    )


if __name__ == "__main__":
    input_file = "/home/jeel00dev/Projects/chess_engine_project/data/raw/1.pgn.zst"
    output_dir = "/home/jeel00dev/Projects/chess_engine_project/data/processed/"
    parse_pgn_zst(input_file, output_dir)
