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
parser.add_argument("-e", "--eval_only", type=bool, default=False, help="Write all samples in rool_valid.bin")
#parser.add_argument("-s", "--shard_size", type=int, default=10**8, help="Size of each data shard in the output .bin files, in tokens")
args = parser.parse_args()

random.seed(args.seed)
shard_size = 2**18
name = "rook"
local_dir = "rook"
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)

ds = []
for fn in glob(args.input):
    with open(fn, "r") as f:
        ds += [l for l in f.readlines() if l.strip()]

# remove exact duplicates & shuffle
# TODO: maybe remove close duplicates
#  - e.g. only difference is randomized move order
#  - or slightly different eval score for one move
# TODO: dont do this in-memory for large datasets

ds = list(set(ds))
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
            write_datafile(filename, all_tokens_np)
            shard_index += 1
            progress_bar = None
            # populate the next shard with the leftovers of the current doc
            all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
            token_count = len(tokens)-remainder

    # write any remaining tokens as the last shard
    if token_count != 0:
        split = "val" if args.eval_only else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"{name}_{split}_{shard_index:06d}.bin")
        write_datafile(filename, all_tokens_np[:token_count])
