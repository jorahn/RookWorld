import os, random, psutil, argparse

import chess
import chess.engine
from datasets import load_dataset
from tqdm import tqdm

parser = argparse.ArgumentParser(description="ROOK eval dataset generation")
parser.add_argument("-p", "--stockfish_path", type=str, help="Path to the stockfish binary")
parser.add_argument("-o", "--output", type=str, default="rook_eval.txt", help="Filename of the generated output dataset")
parser.add_argument("-l", "--timelimit", type=float, default=30, help="Stockfish analysis time limit per position")
parser.add_argument("-t", "--threads", type=int, default=-1, help="Number of threads to use for stockfish analysis. -1 = for one thread per CPU core.")
args = parser.parse_args()

# instantiate the stockfish engine, configure it to use all but one core, fail if the stockfish path is not provided
engine = chess.engine.SimpleEngine.popen_uci(args.stockfish_path)
threads = max(1, psutil.cpu_count(logical=False)) if args.threads == -1 else args.threads
engine.configure({"Threads": threads})
print(f"running stockfish on {threads} threads in parallel")

# load source dataset, either the full dataset or a slice, fail if the dataset is not found or slice invalid
ds = []
with open("eval_fens_1000.txt", "r") as f:
    for line in f:
        ds.append(line)

# generate the stockfish eval dataset
def get_stockfish_analysis(fen, time_limit=0.1):
    board = chess.Board(fen)
    
    # Get top 5 moves
    info = engine.analyse(board, chess.engine.Limit(time=time_limit), multipv=5)
    
    moves = [str(entry["pv"][0]) for entry in info if "pv" in entry]
    evals = [str(entry["score"].relative.score(mate_score=100000) / 100) for entry in info if "score" in entry]
    # moves ordered from high eval (good for white) to low eval (good for black)
    # -> score range -999.99 to 999.99

    return moves, evals

# format the target dataset samples
def format_sample(fen, moves, evals):
    is_black_turn = fen.split()[1] == "b"

    # Randomize move order
    combined = list(zip(moves, evals))
    random.shuffle(combined)
    shuffled_moves, shuffled_evals = zip(*combined)
    
    # select best move (first index for white, last for black)
    best_move = moves[-1] if is_black_turn else moves[0]
    
    # format parts
    # TODO: test padding vs packing
    fen_part   = f"P: {fen:90}" # https://chess.stackexchange.com/questions/30004/longest-possible-fen
    moves_part = f"M: {' '.join(shuffled_moves):30}"
    evals_part = f"E: {' '.join(shuffled_evals):40}"
    best_move  = f"B: {best_move}"
    
    return fen_part + moves_part + evals_part + best_move

# iterate over input dataset, generate stockfish evals, and write to output file
# if output file exists, add "_1" suffix
if os.path.exists(args.output):
    fn, ext = os.path.splitext(args.output)
    args.output = fn + "_1" + ext

# dont parallelize this with dataset.map, as stockfish is already parallelized
with open(args.output, "w") as f:
    for fen in tqdm(ds):
        moves, evals = get_stockfish_analysis(fen, time_limit=args.timelimit)
        formatted_sample = format_sample(fen, moves, evals)
        f.write(formatted_sample+"\n")

engine.quit()
