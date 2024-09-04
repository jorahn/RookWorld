# Run Google BigBench Checkmate in One Benchmark

import argparse, json, io, os

# suppress tensorflow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from transformers import pipeline
import torch
import chess
import chess.pgn
from tqdm import tqdm
import requests

torch.manual_seed(42)

parser = argparse.ArgumentParser(description="ROOK accuracy evaluation")
parser.add_argument("-m", "--model_path", type=str, help="Path to Hugging Face model")
parser.add_argument("-b", "--batch_size", type=int, default=1, help="Inference batch size - recommended to use 1 for this benchmark")
parser.add_argument("-g", "--greedy", action="store_true", help="Use greedy decoding")
parser.add_argument("-t", "--temp", type=float, default=0.6, help="Sampling temperature")
parser.add_argument("-k", "--topk", type=int, default=5, help="Sampling top-k")
parser.add_argument("-n", "--num_samples", type=int, help="Max number of samples to evaluate")
args = parser.parse_args()


URL = "https://github.com/google/BIG-bench/raw/main/bigbench/benchmark_tasks/checkmate_in_one/task.json"

def download():
    """Downloads BIG-bench Checkmate in One"""
    DATA_CACHE_DIR = "../data/bb-cio"
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    data_filename = os.path.join(DATA_CACHE_DIR, f"task.json")
    if not os.path.exists(data_filename):
        print(f"Downloading {URL} to {data_filename}...")
        with open(data_filename, "wb") as f:
            f.write(requests.get(URL).content)
    else:
        print(f"{data_filename} already exists, skipping download...")

if args.batch_size > 1:
    p = pipeline("text-generation", model=args.model_path, device_map="auto", torch_dtype=torch.bfloat16, batch_size=args.batch_size)
    p.tokenizer.padding_side = "left"
    p.tokenizer.pad_token_id = p.model.config.eos_token_id
    
    print("WARNING: generating with batch_size > 1 leads to slightly worse results")
    # with batch_size > 1, the model will generate different completions for the same prompt, vs batch_size=1
    # even when setting torch.manual_seed()
    # this generally reduces the accurady in this benchmark, need to investigate further
    # https://huggingface.co/docs/transformers/generation_strategies#greedy-search
    # could this be related to padding in batches?
    # for now, use batch_size=1
    # TODO check if left padding solves this
else:
    p = pipeline("text-generation", model=args.model_path, device="cuda", torch_dtype=torch.bfloat16)

# generate completion after FEN
def generate_completion(fens):
    prompt = ["P: "+fen for fen in fens]
    gen_args = {
        "max_length": 200,
        "truncation": True,
        "pad_token_id": p.tokenizer.eos_token_id,
        }
    if args.greedy:
        res = p(prompt, do_sample=False, **gen_args)
    else:
        res = p(prompt, do_sample=True, temperature=args.temp, top_k=args.topk, **gen_args)
    gen = [r[0]["generated_text"] for r in res]
    return gen

def load_task():
    with open("../data/bb-cio/task.json", "r") as f:
        data = json.load(f)
    if args.num_samples:
        return data["examples"][:args.num_samples]
    return data["examples"]

def run_task():
    data = load_task()
    targets, predictions = [], []
    total_batches = (len(data) + args.batch_size - 1) // args.batch_size
    for i in tqdm(range(0, len(data), args.batch_size), total=total_batches, desc="Evaluating"):
        batch = data[i:i+args.batch_size]
        games = [chess.pgn.read_game(io.StringIO(ex["input"])) for ex in batch]
        boards = [game.board() for game in games]
        for game, board in zip(games, boards):
            for move in game.mainline_moves(): board.push(move)        
        fens = [board.fen() for board in boards]
        target_moves = [board.parse_san(ex["target"]) for ex, board in zip(batch, boards)]
        targets += target_moves

        predicted_moves = []
        for g in generate_completion(fens):
            try:
                gen = g.split("B:", 1)[1].split()[0].strip()
                predicted_moves.append(chess.Move.from_uci(gen))
            except (IndexError, chess.InvalidMoveError, AssertionError):
                predicted_moves.append(None)
        predictions += predicted_moves

    return targets, predictions

print("Evaluating ROOK model on Google Big Bench Checkmate in One Task")
targets, predictions = run_task()
correct = sum(t == p for t, p in zip(targets, predictions))
print(f"Accuracy: {correct/len(targets):.2%} ({correct}/{len(targets)})")
