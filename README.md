# ROOK: Reasoning Over Organized Knowledge

A chess-playing transformer model trained on a synthetic dataset with chain-of-thought evaluation from Stockfish.

## setup
- download and unpack stockfish binary from [here](https://github.com/official-stockfish/Stockfish)
- create and activate a clean environment
- clone this repo `git clone https://github.com/jorahn/rook.git`
- cd into the repo folder and run `setup.sh`
  - this will clone llm.c at a specific commit
  - copies files for dataset generation, training and evaluation of ROOK into llm.c
- finalize environment setup for llm.c with dependencies like cuda, cudnn and nccl as per llm.c docs

## generate dataset
1. generate a text-dataset with stockfish (very cpu intense)
  1.1 to generate a text-dataset from human chess positions run `llm.c/dev/data/rook/generate_lichess.py`
  1.2 to generate a text-dataset from stockfish self-play positions run `llm.c/dev/data/rook/generate_selfplay.py`
2. to generate llm.c train and valid files (.bin) from a text-dataset run `llm.c/dev/data/rook.py`

## run training
- modify / run `llm.c/scripts/run_gpt2_124M_rook.sh`
- for monitoring, run `jupyter lab` in `llm.c/dev/` and open `vislog2_rook.ipynb`

## evaluation
- run `llm.c/dev/eval/export_hf.py` on a model.bin
- run `llm.c/dev/eval/rook_self_play.py` to play the converted model against itself, observe number of moves before illegal move
- run `llm.c/dev/eval/rook_analysis.py` to provide an FEN (e.g. from a human game) and get the model evaluation for it
