import argparse, random

# suppress tensorflow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from transformers import pipeline
import torch
import chess
import chess.engine
import psutil
from tqdm import tqdm

parser = argparse.ArgumentParser(description="ROOK self-play evaluation for illegal moves")
parser.add_argument("-m", "--model_path", type=str, help="ROOK hf gpt2 model")
parser.add_argument("-n", "--num_games", type=int, default=50, help="number of selfplay games")
parser.add_argument("-g", "--greedy", action="store_true", help="Use greedy decoding")
parser.add_argument("-tp", "--temp", type=float, default=0.6, help="Sampling temperature")
parser.add_argument("-k", "--topk", type=int, default=5, help="Sampling top-k")
parser.add_argument("-p", "--stockfish_path", type=str, help="Path to the stockfish binary")
parser.add_argument("-s", "--stockfish_level", type=int, default=0, help="Stockfish level (0-20)")
parser.add_argument("-l", "--timelimit", type=float, default=0.1, help="Stockfish analysis time limit per position")
parser.add_argument("-t", "--threads", type=int, default=-1, help="Number of threads to use for stockfish analysis. -1 = for one thread per CPU core.")
args = parser.parse_args()

print("ROOK model win/loss/draw evaluation against Stockfish")

# instantiate the stockfish engine, configure it to use all but one core, fail if the stockfish path is not provided
engine = chess.engine.SimpleEngine.popen_uci(args.stockfish_path)
threads = max(1, psutil.cpu_count(logical=False)) if args.threads == -1 else args.threads
engine.configure({"Threads": threads, "Skill Level": args.stockfish_level})
print(f"running Stockfish 16.1 on level {args.stockfish_level} on {threads} threads in parallel")

if args.greedy:
    print("Greedy decoding - this will result in the same moves played for every game")

device = "cuda" if torch.cuda.is_available() else "cpu"
p = pipeline("text-generation", model=args.model_path, 
             device=device, torch_dtype=torch.bfloat16)

def stockfish_play(board):
    result = engine.play(board, chess.engine.Limit(time=args.timelimit))
    return result.move

def rook_play(board):
    if args.greedy:
        gen = p("P: "+board.fen(), max_length=256, truncation=True, 
                do_sample=False, pad_token_id=p.tokenizer.eos_token_id)
    else:
        gen = p("P: "+board.fen(), max_length=256, truncation=True, temperature=args.temp, 
                top_k=args.topk, do_sample=True, pad_token_id=p.tokenizer.eos_token_id)
    move = gen[0]["generated_text"].split()[-1]
    return chess.Move.from_uci(move)

counters = []
results = []
for _ in tqdm(range(args.num_games)):
    counter = 0
    board = chess.Board()
    rook_white = random.choice([True, False])

    err = False
    while not board.is_game_over():
        if board.turn == chess.WHITE and rook_white or board.turn == chess.BLACK and not rook_white:
            move = rook_play(board)
            if move not in board.legal_moves:
                print("Game ended with illegal move:", move, board.fen())
                err = True
                if rook_white: results.append(("0-1", rook_white))
                else: results.append(("1-0", rook_white))
                break
            board.push(move)
        else:
            move = stockfish_play(board)
            board.push(move)
        counter += 1
    counters.append(counter)
    if not err:
        print("Game completed with result:", board.result())
        results.append((board.result(), rook_white))

print("Average game length:", sum(counters) / len(counters))
print("Max game length:", max(counters))
print("Min game length:", min(counters))
print("Results:", results)

wld = {"rook_white": 0, "rook_black": 0}
for r in results:
    res, rook_white = r
    if res == "1-0" and rook_white:
        wld["rook_white"] += 1
    elif res == "0-1" and not rook_white:
        wld["rook_black"] += 1
    elif res == "1/2-1/2":
        if rook_white:
            wld["rook_white"] += 0.5
        else:
            wld["rook_black"] += 0.5
print("WLD:", wld)
print("ROOK winrate:", (wld["rook_white"] + wld["rook_black"]) / args.num_games)
engine.quit()