<img src="logo.png" width="150" height="150">

# ROOK: Reasoning Over Organized Knowledge

A chess-playing transformer model trained on a synthetic dataset with chain-of-thought evaluation from Stockfish.

## setup
- tested: linux (ubuntu 22.04), python 3.11, nvidia-gpu, cuda 12.4, cudnn 9.3, stockfish 16.1
- download and unpack stockfish binary from [here](https://github.com/official-stockfish/Stockfish)
- set env STOCKFISH_PATH to binary with `export $STOCKFISH_PATH=/path/to/stockfish-binary`
- create and activate a clean python virtual environment / conda environment
- clone this repo `git clone https://github.com/jorahn/rook.git`
- cd into the repo folder `cd rook` and run `bash setup.sh`
  - this will clone llm.c at a specific commit
  - copies files for dataset generation, training and evaluation of ROOK into llm.c
- finalize environment setup for llm.c with dependencies like cuda, cudnn or cudnn-frontend and nccl as per llm.c docs
- `bash run.sh` for 
  - basic data gen (~20k samples, half human and selfplay, ~30 mins on 6 cores)
  - train minimal model on one GPU for 5000 steps (2 epochs) with bs=1 to val-loss ~0.73
  - convert model.bin to hf and run self-play eval (avg ~3.5 legal moves)

### data scaling

| Samples | Steps/Epochs | Val-Loss | Selfplay Legal Moves |
|---------|--------------|----------|----------------------|
|    20k  |    5000 / 2  |   0.73   |          3.5         |
|   260k  |   18624 / 1  |   0.56   |         15.5         |
|   709k  |   51481 / 1  |   0.59   |         19.2         |

*different val-data, work in progress
**comparisons: 14 moves after 2.4m examples [here](https://slatestarcodex.com/2020/01/06/a-very-unlikely-chess-game/)
<img src="yolo.png" width="585" height="662">


## generate dataset
1. generate a text-dataset with stockfish (very cpu intense)
   1. to generate a text-dataset from human chess positions run `llm.c/dev/data/rook/generate_lichess.py -p $STOCKFISH_PATH`
   2. to generate a text-dataset from stockfish self-play positions run `llm.c/dev/data/rook/generate_selfplay.py -p $STOCKFISH_PATH`
3. to generate llm.c train and valid files (.bin) from text-datasets run `llm.c/dev/data/rook.py`

## run training
- modify / run `llm.c/scripts/run_gpt2_124M_rook.sh`
- for monitoring, run `jupyter lab` in `llm.c/dev/` and open `vislog2_rook.ipynb`

## evaluation
- run `llm.c/dev/eval/export_hf.py` to convert model.bin to huggingface gpt2 safetensor + tokenizer
- run `llm.c/dev/eval/rook_selfplay.py` to play the converted model against itself, observe number of moves before illegal move
- run `llm.c/dev/eval/rook_analysis.py` to provide an FEN (e.g. from a human game) and get the model evaluation for it
