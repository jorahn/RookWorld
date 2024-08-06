import chess
import chess.engine
import random
import sys
from datasets import load_dataset
from tqdm import tqdm
import psutil

engine = chess.engine.SimpleEngine.popen_uci(sys.argv[2])
threads = max(1, psutil.cpu_count(logical=False) - 1)
engine.configure({"Threads": threads})
print(f"running stockfish on {threads} threads in parallel")

def get_stockfish_analysis(fen, depth=20):
    board = chess.Board(fen)
    
    # Get top 5 moves
    info = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=5)
    
    moves = [str(entry["pv"][0]) for entry in info]
    evals = [str(entry["score"].relative.score(mate_score=100000) / 100) for entry in info]
    
    return moves, evals

def format_sample(fen, moves, evals):
    is_black_turn = fen.split()[1] == "b"

    # Randomize move order
    combined = list(zip(moves, evals))
    random.shuffle(combined)
    shuffled_moves, shuffled_evals = zip(*combined)
    
    # select best move (first for white, last for black)
    best_move = moves[-1] if is_black_turn else moves[0]
    # Format parts
    # test padding vs packing
    fen_part   = f"P: {fen:90}" # https://chess.stackexchange.com/questions/30004/longest-possible-fen
    moves_part = f"M: {' '.join(shuffled_moves):30}"
    evals_part = f"E: {' '.join(shuffled_evals):40}"
    best_move  = f"B: {best_move}"
    
    return fen_part + moves_part + evals_part + best_move

ds = load_dataset("jrahn/yolochess_lichess-elite_2211")
max_n = int(sys.argv[1])

with open("ds.txt", "w") as f:
    for n, fen in enumerate(tqdm(ds["train"]["fen"])):
        moves, evals = get_stockfish_analysis(fen)
        formatted_sample = format_sample(fen, moves, evals)
        f.write(formatted_sample+"\n")

        if n > max_n: break

engine.quit()
