"""
ChessReason dataset

example doc to highlight the structure of the dataset (newline delimited text file):

P: r1b1kbnr/pp2pppp/n1p5/q2PP3/3P4/2N5/PP3PPP/R1BQKBNR b KQkq - 2 6                          M: c6d5 a6b4 c8d7 a6c7 a6b8      E: -2.0 -2.07 -2.92 -2.79 -2.66            B: c8d7
"""

import os, argparse, random
import multiprocessing as mp
from glob import glob

import numpy as np
import tiktoken
from tqdm import tqdm

from data_common import write_datafile
# ------------------------------------------

parser = argparse.ArgumentParser(description="ROOK dataset preprocessing")
parser.add_argument("-i", "--input", type=str, default="rook/rook_*.txt", help="ROOK dataset version txt-files")
parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed")
parser.add_argument("-e", "--eval_only", type=bool, default=False, help="Create fixed validation set rook_val_500.bin")
parser.add_argument("-c", "--clear", type=str, default="", help="Insert '-' instead of actual data into [M|E] fields")
parser.add_argument("-d", "--debug", type=bool, default=False, help="Write to stdout instead of to file")
#parser.add_argument("-s", "--shard_size", type=int, default=10**8, help="Size of each data shard in the output .bin files, in tokens")
args = parser.parse_args()


random.seed(args.seed)
shard_size = 2**18 
# for now only small datasets
# originally reduced to create a suitable small validation set, maybe increase back to 2**20
name = "rook"
local_dir = "rook"
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)

# TODO: dont do this in-memory for large datasets

ds = []
n = 0
for fn in glob(args.input):
    n += 1
    with open(fn, "r") as f:
        ds += [l for l in f.readlines() if l.strip()]
print(f"{len(ds):,} samples extracted from {n} source files")

# make sure all samples contain all parts
ds = [e for e in ds if all(["P: " in e, "M: " in e, "E: " in e, "B: " in e])]
print(f"{len(ds):,} samples after removing invalid samples")

# clear selected fields
if "M" in args.clear or "E" in args.clear:
    for n, e in enumerate(ds):
        if args.debug: print(e)
        try:
            p1, p2 = e.split("M: ", 1)
            p2, p3 = p2.split("E: ", 1)
            p3, p4 = p3.split("B: ", 1)
            if "M" in args.clear: p2 = "-"*len(p2)
            if "E" in args.clear: p3 = "-"*len(p3)
            ds[n] = f"{p1}M: {p2}E: {p3}B: {p4}"
            if args.debug: print(ds[n])
        except Exception as err:
            print("failed processing", e)
            print("error", err)


# remove exact duplicates
ds = set(ds)
print(f"{len(ds):,} samples after exact deduplication")

# TODO: maybe remove close duplicates
#  - e.g. only difference is randomized move order
#  - or slightly different eval score for one move

# avoid eval contamination
if not args.eval_only:
    with open("rook/rook_val_500.txt.bak", "r") as f:
        ds_val = set([l.strip() for l in f.readlines()])
    ds = ds - ds_val
    print(f"{len(ds):,} samples after eval decontamination")

# shuffle
ds = list(ds)
random.shuffle(ds)


# init the tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>'] # end of text token
def tokenize(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    tokens = [eot] # the special <|endoftext|> token delimits all documents
    tokens.extend(enc.encode_ordinary(doc))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

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
            split = "val" if args.eval_only else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"{name}_{split}_{shard_index:06d}.bin")
            # split the document into whatever fits in this shard; the remainder goes to next one
            remainder = shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
            if args.debug: print(all_tokens_np) 
            else: write_datafile(filename, all_tokens_np)
            shard_index += 1
            progress_bar = None
            # populate the next shard with the leftovers of the current doc
            all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
            token_count = len(tokens)-remainder

    # write any remaining tokens as the last shard
    if token_count != 0:
        split = "val" if args.eval_only else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"{name}_{split}_{shard_index:06d}.bin")
        if args.debug: print(all_tokens_np[:token_count])
        else: write_datafile(filename, all_tokens_np[:token_count])
