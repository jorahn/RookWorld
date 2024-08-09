import sys

from transformers import pipeline
import torch
import chess
from tqdm import tqdm

p = pipeline("text-generation", model=sys.argv[1], device="cuda", torch_dtype=torch.bfloat16)
counters = []
for n in tqdm(range(50)):
    counter = 0
    board = chess.Board()
    err = False
    while not board.is_game_over():
        #print(board)
        gen = p("P: "+board.fen(), max_length=256, truncation=True, temperature=0.6, 
                top_k=5, do_sample=True, pad_token_id=p.tokenizer.eos_token_id)
        move = gen[0]["generated_text"].split()[-1]
        #print("Move:", move)
        try:
            move = chess.Move.from_uci(move)
            if move not in board.legal_moves:
                print("Game ended with illegal move:", move, board.fen())
                print(gen[0]["generated_text"])
                err = True
                break
            board.push(move)
            counter += 1
        except (chess.InvalidMoveError, AssertionError) as e:
            print("Game ended with invalid move:", move, board.fen())
            print(e)
            print(gen[0]["generated_text"])
            err = True
            break
    counters.append(counter)
    if not err:
        print("Game completed with result:", board.result())

print("Average game length:", sum(counters) / len(counters))
print("Max game length:", max(counters))
print("Min game length:", min(counters))
