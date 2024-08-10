from glob import glob
import chess
import random

# extract FENs from preprocessed pgns for Stockfish evaluation
# balance early, middle and late game

stats = {"early": 0, "mid": 0, "end": 0}
fens = set()
for fn in glob("*.pgn"):
    if not "noeco" in fn:
        with open(fn, "r") as f:
            for line in f:
                if line.startswith("{ "):
                    fen = line.replace("{ ", "").replace(" }", "").strip()
                    try:
                        # validate fen
                        chess.Board(fen)
                    
                        move = int(fen.split()[-1])
                        if move < 10: stats["early"] += 1
                        elif move < 30: stats["mid"] += 1
                        else: stats["end"] += 1
                        fens.add(fen)
                    except Exception as e:
                        print(e)
                        print(line)

print(len(fens))
print(stats)

fens = list(fens)
random.shuffle(fens)

with open("eval_fens.txt", "w") as f:
    for fen in fens:
        f.write(fen+"\n")
