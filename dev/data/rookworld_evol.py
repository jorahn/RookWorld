"""
Arbiter dataset

example doc to highlight the structure of the dataset (newline delimited text file):

5R2/6R1/8/3P4/p7/1b2R2P/2p3P1/6K1 b - - 0 58+b3d5+e1e3 a2b3 f7f2 b5b4 g5g7 b4c3 f2f8 c3c2 d4d5 b3d5+5R2/6R1/8/3b4/p7/4R2P/2p3P1/6K1 w - - 0 59+0.001+0+0
"""

import os, argparse, random
import multiprocessing as mp
from glob import glob

import numpy as np
import tiktoken
from tqdm import tqdm
from datasets import load_dataset

from data_common import write_datafile
# ------------------------------------------

parser = argparse.ArgumentParser(description="RookWorld Evol dataset preprocessing")
parser.add_argument("-i", "--input", type=str, default="jrahn/rookworld_evol_st1_800k", help="RookWorld Evol dataset")
parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed")
args = parser.parse_args()


random.seed(args.seed)
shard_size = 2**24 
name = args.input.split("/")[-1].replace("_", "-")
local_dir = "rookworld_evol"
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

ds = load_dataset(args.input)
print(f"{len(ds['train']):,} train samples, {len(ds['test']):,} test samples")


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

process_dataset(ds["train"], "train")
process_dataset(ds["test"], "val")
