from argparse import ArgumentParser

from datasets import load_dataset, Dataset, concatenate_datasets
import json
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument("-i", "--input", type=str, default="rookworld_selfplay.txt", help="Path to txt-input file")
parser.add_argument("-ad", "--arbiter_dataset", type=str, default="", help="Train environment generation model with ArbiterSim dataset, ignore arbiter data in input file")
parser.add_argument("-t", "--test_size", type=int, default=10_000, help="Size of test split")
parser.add_argument("-p", "--push", type=str, help="repo_id for Push to Hugging Face Hub")
args = parser.parse_args()

samples = set()
with open(args.input, "r") as f:
    for line in f:
        if not (args.arbiter_dataset and line.startswith("A: ")):
            samples.add(line)
        
samples = [{"text": s.strip()} for s in samples]

ds = Dataset.from_list(samples)

if args.arbiter_dataset:
    ds_arbiter = load_dataset(args.arbiter_dataset, split="train")
    ds_arbiter = ds_arbiter.map(lambda x: {"text": f"A: {x['text']} "})
    ds = concatenate_datasets([ds, ds_arbiter])

# shuffle and train-test-split
ds = ds.train_test_split(test_size=args.test_size)

print("processed dataset:")
print(ds)
print(ds["train"][0])
print(ds["train"][1])
print(ds["test"][0])
print(ds["test"][1])

if args.push:
    print("pushing to Hugging Face Hub")
    ds.push_to_hub(args.push)