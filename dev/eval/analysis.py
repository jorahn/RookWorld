import sys

from transformers import pipeline
import torch
import chess
from tqdm import tqdm

p = pipeline("text-generation", model=sys.argv[1], device="cuda", torch_dtype=torch.bfloat16)
fen = input("FEN:")
board = chess.Board(fen)

gen = p("P: "+board.fen(), max_length=256, truncation=True, temperature=0.6, 
                top_k=5, do_sample=True, pad_token_id=p.tokenizer.eos_token_id)
print(gen[0]["generated_text"])
