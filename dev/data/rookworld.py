import os, argparse, random
import multiprocessing as mp
from glob import glob

import numpy as np
import tiktoken
from tqdm import tqdm
from datasets import load_dataset, interleave_datasets

from data_common import write_datafile
# ------------------------------------------


parser = argparse.ArgumentParser(description="RookWorld dataset preprocessing")
parser.add_argument("-dsr", "--dataset_rook", type=str, default="lfsm/rook-5m", help="ROOK dataset")
parser.add_argument("-dsa", "--dataset_arbiter", type=str, default="jrahn/arbiter_2m", help="ARBITER dataset")
parser.add_argument("-o", "--output", type=str, default="jrahn/rookworld_7m", help="RookWorld HF dataset")
parser.add_argument("-p", "--push_to_hub", action="store_true", help="Push to Hugging Face Hub")
parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed")
args = parser.parse_args()


random.seed(args.seed)
shard_size = 2**24 

name = args.output.split("/")[-1].replace("_", "-")
local_dir = "rookworld"
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)


# init the tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>'] # end of text token
def tokenize(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    tokens = [eot] # the special <|endoftext|> token delimits all documents
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

def process_dataset(ds, split):
    # tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
    nprocs = max(1, os.cpu_count() - 2) # don't hog the entire system
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        # preallocate buffer to hold current shard
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0
        progress_bar = None
        for tokens in pool.imap(tokenize, ds, chunksize=16):

            # is there enough space in the current shard for the new tokens?
            if token_count + len(tokens) < shard_size:
                # simply append tokens to current shard
                all_tokens_np[token_count:token_count+len(tokens)] = tokens
                token_count += len(tokens)
                # update progress bar
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(len(tokens))
            else:
                # write the current shard and start a new one
                filename = os.path.join(DATA_CACHE_DIR, f"{name}_{split}_{shard_index:06d}.bin")
                # split the document into whatever fits in this shard; the remainder goes to next one
                remainder = shard_size - token_count
                progress_bar.update(remainder)
                all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
                write_datafile(filename, all_tokens_np)
                shard_index += 1
                progress_bar = None
                # populate the next shard with the leftovers of the current doc
                all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
                token_count = len(tokens)-remainder

        # write any remaining tokens as the last shard
        if token_count != 0:
            filename = os.path.join(DATA_CACHE_DIR, f"{name}_{split}_{shard_index:06d}.bin")
            write_datafile(filename, all_tokens_np[:token_count])


ds1 = load_dataset(args.dataset_rook, split="train")
ds2 = load_dataset(args.dataset_arbiter, split="train")
len1 = len(ds1)
len2 = len(ds2)
total_len = len1 + len2

ds = interleave_datasets([ds1, ds2], probabilities=[len1/total_len, len2/total_len])
ds = ds.train_test_split(test_size=15_000, seed=args.seed)
print(ds)

if args.push_to_hub:
    ds.push_to_hub(args.output)
print(f"{len(ds['train']):,} train samples, {len(ds['test']):,} test samples")

process_dataset(ds["train"], "train")
process_dataset(ds["test"], "val")

