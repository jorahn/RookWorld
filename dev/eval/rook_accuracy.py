import argparse

# suppress tensorflow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from transformers import pipeline
from datasets import load_dataset
import torch
import chess
from tqdm import tqdm

parser = argparse.ArgumentParser(description="ROOK accuracy evaluation")
parser.add_argument("-m", "--model_path", type=str, help="Path to Hugging Face model")
parser.add_argument("-d", "--data_path", type=str, help="Path to validation dataset (.txt)")
parser.add_argument("-g", "--greedy", action="store_true", help="Use greedy decoding")
parser.add_argument("-t", "--temp", type=float, default=0.6, help="Sampling temperature")
parser.add_argument("-k", "--topk", type=int, default=5, help="Sampling top-k")
parser.add_argument("-b", "--batch_size", type=int, default=1, help="Batch size")
args = parser.parse_args()

p = pipeline("text-generation", model=args.model_path, device_map="auto", torch_dtype=torch.bfloat16, batch_size=args.batch_size)
p.tokenizer.pad_token_id = p.model.config.eos_token_id
p.tokenizer.padding_side = "left"

def process_rook_format(sample):
    # split validation sample into parts Postion, Moves, Evals, Best move
    sample_p, sample_m = sample.split("M: ", 1)
    sample_m, sample_e = sample_m.split("E: ", 1)
    sample_e, sample_b = sample_e.split("B: ", 1)
    fen = sample_p.split("P: ")[1].strip()
    sample_moves = [e.strip() for e in sample_m.strip().split()]
    sample_evals = [float(e.strip()) for e in sample_e.strip().split()]
    sample_best = sample_b.strip()
    return fen, sample_moves, sample_best

# TODO convert to HF dataset
if args.data_path.endswith(".txt"):
    with open(args.data_path, "r") as f:
        ds = [l for l in f.readlines() if l.strip()]
elif args.data_path.endswith(".csv"):
    ds = load_dataset("csv", data_files=args.data_path)["train"]
    ds = ds.select(range(1000))


print("evaluating ROOK model on validation dataset (illegal moves and accuracy)")

stats = {
    "total": 0, 
    "best_move_legal": [],
    "best_move_correct": [],
    "top_5_legal": [],
    "top_5_correct": [],
    "invalid_completion": 0,
    "invalid_completion_b": 0,
    "invalid_completion_m": 0,
}
for sample in tqdm(ds):
    stats["total"] += 1

    if isinstance(sample, str):
        fen, sample_moves, sample_best = process_rook_format(sample)
    else:
        fen = sample["FEN"]
        sample_moves = [""] * 5
        sample_best = sample["Move"]
    
    # generate completion after FEN
    if args.greedy:
        res = p("P: "+fen, max_length=256, truncation=True, 
                do_sample=False, pad_token_id=p.tokenizer.eos_token_id)
    else:
        res = p("P: "+fen, max_length=256, truncation=True, temperature=args.temp, 
                    top_k=args.topk, do_sample=True, pad_token_id=p.tokenizer.eos_token_id)
    gen = res[0]["generated_text"]

    invalid = False
    # check if generated completion contains "B: " part
    if "B: " in gen:
        gen_best = gen.split("B: ", 1)[1].strip()

        # check if best move is legal
        board = chess.Board(fen)
        try:
            stats["best_move_legal"].append(chess.Move.from_uci(gen_best) in board.legal_moves)
        except chess.InvalidMoveError:
            stats["best_move_legal"].append(False)
            invalid = True
            stats["invalid_completion_b"] += 1

        # check if best move is correct
        stats["best_move_correct"].append(sample_best == gen_best)
    else:
        invalid = True
        stats["invalid_completion_b"] += 1

    # check if generated completion contains "M: " part
    if "M: " in gen and "E: " in gen:
        gen_m = gen.split("M: ", 1)[1].split("E: ", 1)[0].strip()
        gen_moves = [e.strip() for e in gen_m.strip().split()]
        try:
            gen_moves_chess = [chess.Move.from_uci(m) for m in gen_moves]

            board = chess.Board(fen)
            # check how many of the top 5 moves are legal
            stats["top_5_legal"].append(sum([m in board.legal_moves for m in gen_moves_chess]))

            # check how many of the top 5 moves are correct
            top_5_correct = len(set(sample_moves) & set(gen_moves))
            stats["top_5_correct"].append(top_5_correct)
        except chess.InvalidMoveError:
            stats["top_5_legal"].append(0)
            invalid = True
            stats["invalid_completion_m"] += 1

    else:
        invalid = True
        stats["invalid_completion_m"] += 1

    if invalid:
        stats["invalid_completion"] += 1

# print results
print(f"Total samples: {stats['total']}")
print(f"Invalid completions: {stats['invalid_completion']/stats['total']:.2%}")
print(f"Best move legal: {sum(stats['best_move_legal'])/stats['total']:.2%}")
print(f"Best move accuracy: {sum(stats['best_move_correct'])/stats['total']:.2%}")
print(f"Top 5 move legal: {sum(stats['top_5_legal'])/5/stats['total']:.2%}")
print(f"Top 5 move accuracy: {sum(stats['top_5_correct'])/5/stats['total']:.2%}")

