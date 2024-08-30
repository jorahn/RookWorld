import argparse

from transformers import pipeline
import torch
import chess
from tqdm import tqdm

parser = argparse.ArgumentParser(description="ROOK self-play evaluation for illegal moves")
parser.add_argument("-m", "--model_path", type=str, help="ROOK hf gpt2 model")
parser.add_argument("-n", "--num_games", type=int, default=50, help="number of selfplay games")
parser.add_argument("-g", "--greedy", action="store_true", help="Use greedy decoding")
parser.add_argument("-t", "--temp", type=float, default=0.6, help="Sampling temperature")
parser.add_argument("-k", "--topk", type=int, default=5, help="Sampling top-k")
parser.add_argument("-b", "--batch_size", type=int, default=1, help="Batch size")
args = parser.parse_args()

print("ROOK model self-play evaluation for illegal moves and completed games")

if args.greedy:
    print("Greedy decoding - this will result in the same moves played for every game")

p = pipeline("text-generation", model=args.model_path, device_map="auto", torch_dtype=torch.bfloat16, batch_size=args.batch_size)
p.tokenizer.pad_token_id = p.model.config.eos_token_id
p.tokenizer.padding_side = "left"

counters = []
for _ in tqdm(range(args.num_games)):
    counter = 0
    board = chess.Board()
    err = False
    while not board.is_game_over():
        
        if args.greedy:
            gen = p("P: "+board.fen(), max_length=256, truncation=True, 
                    do_sample=False, pad_token_id=p.tokenizer.eos_token_id)
        else:
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
