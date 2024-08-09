cd llm.c/dev/data/rook
#python generate_lichess.py -p $STOCKFISH_PATH
#python generate_selfplay.py -p $STOCKFISH_PATH
cd ..

#python rook.py
cd ../..

#bash scripts/run_gpt2_124M_rook.sh

cd dev/eval
python export_hf.py --input ../../log_gpt2_124M_rook/model_00002508.bin --output ../../log_gpt2_124M_rook/rook_gpt2_124M_hf

python rook_selfplay.py --model_path ../../log_gpt2_124M_rook/rook_gpt2_124M_hf/
