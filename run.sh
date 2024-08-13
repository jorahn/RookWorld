cd llm.c/dev/data/rook
python generate_lichess.py -p $STOCKFISH_PATH
python generate_selfplay.py -p $STOCKFISH_PATH
cd ..

python rook.py
cd ../..

bash scripts/run_gpt2_124M_rook.sh

cd dev/eval
python export_hf.py --input ../../log_gpt2_124M_rook/model_00005000.bin --output ../../log_gpt2_124M_rook/rook_gpt2_124M_hf

# run Google Big Bench Checkmate in One Benchmark
python rook_bb-cio.py -g --model_path ../../log_gpt2_124M_rook/rook_gpt2_124M_hf/

# evaluate the model on the validation set (legal move %, best move accuracy, top 5 move accuracy)
python rook_accuracy.py -g --model_path ../../log_gpt2_124M_rook/rook_gpt2_124M_hf/ -d ../data/rook/rook_val_500.txt.bak

# play 50 games with greedy decoding vs stockfish at strength level 0 and 50ms time limit per move
python rook_vs_stockfish.py -n 50 -g --model_path ../../log_gpt2_124M_rook/rook_gpt2_124M_hf/ -p $STOCKFISH_PATH -s 0 -l 0.05

# play 50 games against itself with sampling (topk=5, temp=0.6) until illegal move or win/loss/draw outcome
python rook_selfplay.py --model_path ../../log_gpt2_124M_rook/rook_gpt2_124M_hf/

