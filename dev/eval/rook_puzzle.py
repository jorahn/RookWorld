# Lichess Puzzle Database data format:
#PuzzleId                                                       00008
#FEN                r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2R1/PqP2bPP/7K b - ...
#Moves                                  f2g3 e6e7 b2b1 b3c1 b1c1 h6c1
#Rating                                                          1853
#RatingDeviation                                                   76
#Popularity                                                        94
#NbPlays                                                         6405
#Themes                         crushing hangingPiece long middlegame
#GameUrl                        https://lichess.org/787zsVup/black#48
#OpeningTags                                                      NaN

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from argparse import ArgumentParser

from transformers import pipeline
import torch
import chess
from tqdm import tqdm
import pandas as pd
import requests

URL = "https://database.lichess.org/lichess_db_puzzle.csv.zst"
DEBUG = False

parser = ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="jrahn/rook_5m_3e_gpt2_124M_hf")
parser.add_argument("--rw", action="store_true", help="Use RookWorld environment")
parser.add_argument("-n", "--num_puzzles", type=int, default=10_000)
args = parser.parse_args()

pipe = pipeline(
    "text-generation", 
    model=args.model,
    tokenizer=args.model,
    torch_dtype=torch.bfloat16,
    device=0 if torch.cuda.is_available() else -1,
)

gen_settings = {
    "do_sample": False,
    "max_length": 200,
    "truncation": True,
    "return_full_text": False,
    "pad_token_id": pipe.tokenizer.eos_token_id,
}

if not os.path.exists("lichess_db_puzzle.csv.zst"):
    print("Downloading Lichess Puzzle Database")
    with open("lichess_db_puzzle.csv.zst", "wb") as f:
        f.write(requests.get(URL).content)
else:
    print("Lichess Puzzle Database already exists, skipping download")
data = pd.read_csv("lichess_db_puzzle.csv.zst")
print(f"Sampling {args.num_puzzles} puzzles")
data = data.sample(args.num_puzzles, random_state=42)

print("Lichess Puzzle Rating Distribution:")
print(pd.cut(data["Rating"], bins=12).value_counts().sort_index())



def make_move(fen):
    prompt = f"P: {fen} "
    generation = pipe(prompt, **gen_settings)
    try:
        move = generation[0]["generated_text"].split("B: ")[-1].strip()
    except IndexError:
        move = "0000"
    return move
    
stats = {"correct_moves": 0, "solved_puzzles": 0}
for i, row in tqdm(data.iterrows(), total=len(data)):
    targets = row["Moves"].split()

    if args.rw:
        # RookWorld Environment
        raise NotImplementedError("RookWorld Environment not implemented")
    else:
        board = chess.Board(row["FEN"])
        for i, target in enumerate(targets):
            move = make_move(board.fen())
            if DEBUG: print(f"Move: {move}, Target: {target}, FEN: {board.fen()}, i: {i}")
            if move == target:
                stats["correct_moves"] += 1
                if i == len(targets) - 1:
                    stats["solved_puzzles"] += 1
                board.push_uci(target)
            if move != target:
                break

print(stats)

