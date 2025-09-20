import json
import numpy as np


def cp_to_value(entry):
    if "mate" in entry:
        return 1.0 if entry["mate"] > 0 else -1.0
    cp = entry.get("cp", 0)
    return np.tanh(cp / 400.0)


def parse_jsonl_top_n(jsonl_path, out_path, max_samples=10000, top_n=3):
    data = []
    with open(jsonl_path, "r") as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            obj = json.loads(line)
            fen = obj["fen"]

            evals = obj.get("evals", [])
            if not evals:
                continue

            pvs = evals[0].get("pvs", [])
            moves_list = []

            for pv in pvs[:top_n]:
                line_moves = pv.get("line", "").split()
                if not line_moves:
                    continue
                first_move = line_moves[0]
                value = cp_to_value(pv)
                moves_list.append({"move": first_move, "value": value})

            if not moves_list:
                continue

            avg_value = np.mean([m["value"] for m in moves_list])

            sample = {"fen": fen, "moves": moves_list, "value": avg_value}
            data.append(sample)

    with open(out_path, "w") as out:
        json.dump(data, out, indent=2)

    print(f"Saved {len(data)} samples to {out_path}")


jsonl_path = "/home/jeel00dev/Projects/chess_engine/lichess_db_eval.jsonl"
out_path = "/home/jeel00dev/Projects/chess_engine/parsed_data.json"

parse_jsonl_top_n(jsonl_path, out_path, max_samples=10000, top_n=3)
