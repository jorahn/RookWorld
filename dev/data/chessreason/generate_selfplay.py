import chess
import chess.engine
import random
import sys
from datasets import load_dataset
from tqdm import tqdm
import psutil

engine = chess.engine.SimpleEngine.popen_uci(sys.argv[1])
threads = 1 #max(1, psutil.cpu_count(logical=False) - 1)
engine.configure({"Threads": threads})
print(f"running stockfish on {threads} threads in parallel")

def stockfish_selfplay():
    board = chess.Board()
    while not board.is_game_over() or board.ply() <= 60:
        fen = board.fen()

        # Get top 5 moves
        info = engine.analyse(board, chess.engine.Limit(time=1), multipv=5)
        
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

        board.push_uci(best_move)

        # Format parts
        # test padding vs packing
        fen_part   = f"P: {fen:90}" # https://chess.stackexchange.com/questions/30004/longest-possible-fen
        moves_part = f"M: {' '.join(shuffled_moves):30}"
        evals_part = f"E: {' '.join(shuffled_evals):40}"
        best_move  = f"B: {best_move}"
    
        fen_part + moves_part + evals_part + best_move
        with open("ds_selfplay.txt", "a") as f:
            f.write(fen_part + moves_part + evals_part + best_move + "\n")
        
        #board.push(chess.Move.from_uci(best_move))

for n in tqdm(range(int(sys.argv[2]))):
    stockfish_selfplay()

engine.quit()
