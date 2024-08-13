# Run Google BigBench Checkmate in One Benchmark

import argparse, json, io

from transformers import pipeline
import torch
import chess
import chess.pgn
from tqdm import tqdm

parser = argparse.ArgumentParser(description="ROOK accuracy evaluation")
parser.add_argument("-m", "--model_path", type=str, help="Path to Hugging Face model")
parser.add_argument("-g", "--greedy", action="store_true", help="Use greedy decoding")
parser.add_argument("-t", "--temp", type=float, default=0.6, help="Sampling temperature")
parser.add_argument("-k", "--topk", type=int, default=5, help="Sampling top-k")
args = parser.parse_args()

p = pipeline("text-generation", model=args.model_path, device="cuda", torch_dtype=torch.bfloat16)

# generate completion after FEN
def generate_completion(fen):
    if args.greedy:
        res = p("P: "+fen, max_length=256, truncation=True, 
                do_sample=False, pad_token_id=p.tokenizer.eos_token_id)
    else:
        res = p("P: "+fen, max_length=256, truncation=True, temperature=args.temp, 
                    top_k=args.topk, do_sample=True, pad_token_id=p.tokenizer.eos_token_id)
    gen = res[0]["generated_text"]
    return gen

def load_task():
    with open("../data/bb-cio/task.json", "r") as f:
        data = json.load(f)
    return data["examples"]

def run_task():
    stats = {"correct": 0, "total": 0}
    for ex in tqdm(load_task()):
        game = chess.pgn.read_game(io.StringIO(ex["input"]))
        board = game.board()
        for move in game.mainline_moves(): board.push(move)
        target_move = board.parse_san(ex["target"])
        stats["total"] += 1

        try:
            gen = generate_completion(board.fen()).split("B: ", 1)[1].strip()
            predicted_move = chess.Move.from_uci(gen)
        except (AssertionError, chess.InvalidMoveError, IndexError):
            predicted_move = None

        if target_move == predicted_move:
            stats["correct"] += 1
    return stats

print(run_task())