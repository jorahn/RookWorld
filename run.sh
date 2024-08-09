cd llm.c/dev/data/rook
python generate_lichess.py -p $STOCKFISH_PATH
python generate_selfplay.py -p $STOCKFISH_PATH
cd ..

python rook.py
cd ../..

bash scripts/run_gpt2_124M_rook.sh
