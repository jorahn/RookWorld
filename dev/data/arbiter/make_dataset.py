# create a text-dataset to train ArbiterSim environment with causal language modeling

# data format samples from rollouts:
# {"previous_state": "1k1r2nr/1p4b1/2pP3p/q3N1P1/R1PP1B2/3B1N1P/1P3QP1/5RK1 b - - 0 22", "action": "g8e7", "new_state": "1k1r3r/1p2n1b1/2pP3p/q3N1P1/R1PP1B2/3B1N1P/1P3QP1/5RK1 w - - 1 23", "reward": 0.001, "terminated": false, "truncated": false, "recent_moves": ["c1f4", "b4a5", "e4f6", "e6e5", "f6d7", "f8g7", "d7e5", "c8b8", "a1a4", "g8e7"], "_illegal_move": false, "_outcome": "None", "_legal_move_reward": 0.001}
# {"previous_state": "2k5/R7/4p3/1N2P2p/p3b3/P1P5/1P4PP/2K5 b - - 1 47", "action": "h5h4", "new_state": "2k5/R7/4p3/1N2P3/p3b2p/P1P5/1P4PP/2K5 w - - 0 48", "reward": 0.001, "terminated": false, "truncated": false, "recent_moves": ["d6b5", "a6b7", "f3g5", "e7d7", "f6f7", "d7c8", "g5e4", "b7e4", "f7a7", "h5h4"], "_illegal_move": false, "_outcome": "None", "_legal_move_reward": 0.001}

from argparse import ArgumentParser

from datasets import Dataset
import json
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument("-i", "--input", type=str, default="chess_env_rollouts.jsonl", help="Path to jsonl-input file")
parser.add_argument("-t", "--test_size", type=int, default=10_000, help="Size of test split")
parser.add_argument("-p", "--push", type=str, help="repo_id for Push to Hugging Face Hub")
args = parser.parse_args()

samples = set()
counter = 0
with open(args.input, "r") as f:
    for line in f:
        counter += 1
        samples.add(line)
print(f"{counter:,} total lines -> {len(samples):,} unique lines")

samples = [json.loads(s) for s in tqdm(samples, desc="JSON parsing")]
ds = Dataset.from_list(samples)

def format_text(e):
    try:
        e["text"] = f"{e['previous_state']}+{e['action']}+{' '.join(e['recent_moves'])}+{e['new_state']}+{e['reward']}+{int(e['terminated'])}+{int(e['truncated'])}"
    except Exception as ex:
        print("failed processing:", ex)
        print(e)
    return e

ds = ds.filter(lambda x: not None in x["recent_moves"])
ds = ds.map(format_text, num_proc=4)
ds = ds.select_columns("text")

ds = ds.train_test_split(test_size=args.test_size)

print("processed dataset:")
print(ds)

if args.push:
    print("pushing to Hugging Face Hub")
    ds.push_to_hub(args.push)
else:
    for split in ["train", "test"]:
        fn = f"arbitersim_{split}.txt"
        print(f"writing file {fn}")
        with open(fn, "w") as f:
            for e in tqdm(ds["train"], desc=split):
                f.write(e["text"] + "\n")
