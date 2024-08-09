git clone https://github.com/karpathy/llm.c.git
cd llm.c
git checkout -q 6e6a528111cc6641f09d0ebf2ca1e7432d1c87a4
cd ..

cp -r scripts/ llm.c/scripts/
cp -r dev/ llm.c/
pip install -r requirements.txt

cd llm.c
pip install -r requirements.txt

cd dev/
bash download_starter_pack.sh
