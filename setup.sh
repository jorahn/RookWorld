git clone https://github.com/karpathy/llm.c.git
cd llm.c
git checkout -q 6e6a528111cc6641f09d0ebf2ca1e7432d1c87a4
cd ..

pip install chess datasets tqdm

cp -r scripts/ llm.c/scripts/
cp -r dev/ llm.c/

cd llm.c/dev/data/rook
python generate_lichess.py -p $STOCKFISH_PATH
python generate_selfplay.py -p $STOCKFISH_PATH
cd ..

python rook.py
cd ..

bash download_starter_pack.sh
cd ..

bash scripts/run_gpt2_124M_rook.sh
