import random, psutil, argparse

import chess
import chess.engine
from datasets import load_dataset
from tqdm import tqdm

parser = argparse.ArgumentParser(description="ChessReasoner dataset generation from stockfish selfplay")
parser.add_argument("-p", "--stockfish_path", type=str, help="Path to the stockfish binary")
parser.add_argument("-n", "--number_games", type=int, default=500, help="Number of games to play")
parser.add_argument("-o", "--output", type=str, default="chessreason.txt", help="Filename of the generated output dataset")
parser.add_argument("-l", "--timelimit", type=float, default=0.1, help="Stockfish analysis time limit per position")
parser.add_argument("-t", "--threads", type=int, default=-1, help="Number of threads to use for stockfish analysis. -1 = for one thread per CPU core.")
args = parser.parse_args()

# instantiate the stockfish engine, configure it to use all but one core, fail if the stockfish path is not provided
engine = chess.engine.SimpleEngine.popen_uci(args.stockfish_path)
threads = max(1, psutil.cpu_count(logical=False)) if args.threads == -1 else args.threads
engine.configure({"Threads": threads})
print(f"running stockfish on {threads} threads in parallel")

# generate the stockfish eval dataset from selfplay
def stockfish_selfplay(out, time_limit):
    board = chess.Board()
    while not board.is_game_over() or board.ply() <= 60:
        fen = board.fen()

        # Get top 5 moves
        info = engine.analyse(board, chess.engine.Limit(time=time_limit), multipv=5)
        
        # TODO: why does this sometimes fail but not in human games?
        if not all(["pv" in entry for entry in info]): break

        moves = [str(entry["pv"][0]) for entry in info]
        evals = [str(entry["score"].relative.score(mate_score=100000) / 100) for entry in info]
    
        is_black_turn = fen.split()[1] == "b"

        # Randomize move order
        combined = list(zip(moves, evals))
        random.shuffle(combined)
        shuffled_moves, shuffled_evals = zip(*combined)
    
        # select best move (first for white, last for black)
        best_move = moves[-1] if is_black_turn else moves[0]

        # push move before it's modified for formatting
        board.push_uci(best_move)

        # format parts
        # TODO: test padding vs packing
        fen_part   = f"P: {fen:90}" # https://chess.stackexchange.com/questions/30004/longest-possible-fen
        moves_part = f"M: {' '.join(shuffled_moves):30}"
        evals_part = f"E: {' '.join(shuffled_evals):40}"
        best_move  = f"B: {best_move}"
    
        fen_part + moves_part + evals_part + best_move
        out.write(fen_part + moves_part + evals_part + best_move + "\n")
        

with open(args.output, "w") as out:
    for _ in tqdm(range(args.number_games)):
        stockfish_selfplay(out, time_limit=args.timelimit)

engine.quit()
