import argparse

from transformers import pipeline
import torch
import chess

parser = argparse.ArgumentParser(description="ROOK position analysis")
parser.add_argument("-m", "--model_path", type=str, help="Path to Hugging Face model")
parser.add_argument("-g", "--greedy", action="store_true", help="Use greedy decoding")
parser.add_argument("-t", "--temp", type=float, default=0.6, help="Sampling temperature, only used if greedy is False")
parser.add_argument("-k", "--topk", type=int, default=5, help="Sampling top-k, only used if greedy is False")
args = parser.parse_args()

p = pipeline("text-generation", model=args.model_path, device="cuda", torch_dtype=torch.bfloat16)

print("ROOK model manual position analysis")
fen = input("FEN: ")
board = chess.Board(fen)

if args.greedy:
    gen = p("P: "+board.fen(), max_length=256, truncation=True, 
                do_sample=False, pad_token_id=p.tokenizer.eos_token_id)
else:
    gen = p("P: "+board.fen(), max_length=256, truncation=True, temperature=0.6, 
                top_k=5, do_sample=True, pad_token_id=p.tokenizer.eos_token_id)
print(gen[0]["generated_text"])
