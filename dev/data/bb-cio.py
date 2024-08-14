# Download & prepare Google BIG-bench Checkmate in One eval dataset

import multiprocessing as mp
import os, json, io
from glob import glob

import numpy as np
import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel
from tqdm import tqdm

import chess.pgn
import chess

from data_common import download_file, write_evalfile

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "bb-cio")

URL = "https://github.com/google/BIG-bench/raw/main/bigbench/benchmark_tasks/checkmate_in_one/task.json"

enc = tiktoken.get_encoding("gpt2")

def download():
    """Downloads BIG-bench Checkmate in One to DATA_CACHE_DIR"""
    os.makedirs("bb-cio", exist_ok=True)
    data_filename = os.path.join(DATA_CACHE_DIR, f"task.json")
    if not os.path.exists(data_filename):
        print(f"Downloading {URL} to {data_filename}...")
        download_file(URL, data_filename)
    else:
        print(f"{data_filename} already exists, skipping download...")


def render_example(example):
    """
    Given the example as a dictionary, render it as three torch tensors:
    - tokens (the tokens of context + completion, of size 4xN, as there are always 4 candidates)
    - mask (is 1 in the region of the candidate completion, where we evaluate likelihoods)
    - label (the index of the correct completion, which we hope has the highest likelihood)
    """

    """
    HellaSwag example:
    ctx: A man is sitting on a roof. he
    label: 3
    endings: ['is using wrap to wrap a pair of skis.', 'is ripping level tiles off.', "is holding a rubik's cube.", 'starts pulling up roofing on a roof.']
    """
    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]

    # data needed to reproduce this eval on the C size
    data = {
        "label": label,
        "ctx_tokens": None,
        "ending_tokens": [],
    }

    # gather up all the tokens
    ctx_tokens = enc.encode(ctx)
    data["ctx_tokens"] = ctx_tokens
    # TODO generate CoT-continuation from model until "B: "

    tok_rows = []
    mask_rows = []
    for end in endings:
        end_tokens = enc.encode(" " + end) # note: prepending " " because GPT-2 tokenizer
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0]*len(ctx_tokens) + [1]*len(end_tokens))
        data["ending_tokens"].append(end_tokens)

    # have to be careful during the collation because the number of tokens in each row can differ
    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)

    return data, tokens, mask, label

def convert_chess_to_strings(example, multiple_choice_options=4):
    game = chess.pgn.read_game(io.StringIO(example["input"]))
    board = game.board()
    for move in game.mainline_moves(): board.push(move)        
    ctx = board.fen()
    target = board.parse_san(example["target"]).uci()
    
    # select multiple_choice_options - 1 random entries 
    # from example["target_scores"] = list({SAN: score (0|1)})
    # where score == 0 and return them as endings
    endings = []
    for move, score in example["target_scores"].items():
        if score == 0:
            endings.append(board.parse_san(move).uci())
    endings = np.random.choice(endings, min(multiple_choice_options-1, 
                                            len(endings)), replace=False).tolist()
    endings.append(target)
    np.random.shuffle(endings)
    label = endings.index(target)
    return {"ctx": ctx, "label": label, "endings": endings}

def iterate_examples():
    # there are 3,500 examples in total in val
    download()
    with open(os.path.join(DATA_CACHE_DIR, f"task.json"), "r") as f:
        task = json.load(f)
    for ex in task["examples"]:
        if ex:
            yield convert_chess_to_strings(ex)

@torch.no_grad()
def evaluate(model_type, device, n_samples=-1):

    torch.set_float32_matmul_precision('high') # use tf32

    model = GPT2LMHeadModel.from_pretrained(model_type)
    model.to(device)
    # model = torch.compile(model)

    datas = []
    num_correct_norm = 0
    num_correct = 0
    num_total = 0
    for example in iterate_examples():
        if n_samples > 0 and num_total >= n_samples: break
        data, tokens, mask, label = render_example(example)
        datas.append(data)
        tokens = tokens.to(device)
        mask = mask.to(device)

        # get the logits
        logits = model(tokens).logits
        # evaluate the autoregressive loss at all positions
        shift_logits = (logits[..., :-1, :]).contiguous()
        shift_tokens = (tokens[..., 1:]).contiguous()
        flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_shift_tokens = shift_tokens.view(-1)
        shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
        shift_losses = shift_losses.view(tokens.size(0), -1)
        # now get the average loss just for the completion region (where mask == 1), in each row
        shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
        masked_shift_losses = shift_losses * shift_mask
        # sum and divide by the number of 1s in the mask
        sum_loss = masked_shift_losses.sum(dim=1)
        avg_loss = sum_loss / shift_mask.sum(dim=1)
        # now we have a loss for each of the 4 completions
        # the one with the lowest loss should be the most likely
        pred = sum_loss.argmin().item()
        pred_norm = avg_loss.argmin().item()

        # accumulate stats
        num_total += 1
        num_correct += int(pred == label)
        num_correct_norm += int(pred_norm == label)
        print(f"{num_total} acc: {num_correct/num_total:.4f} acc_norm: {num_correct_norm}/{num_total}={num_correct_norm/num_total:.4f}")

        # debug: pretty print a few examples, and the losses in each case
        if num_total < 10:
            print("---")
            print(f"Context:\n {example['ctx']}")
            print(f"Endings:")
            for i, end in enumerate(example["endings"]):
                print(f"{i} (loss: {avg_loss[i].item():.4f}) {end}")
            print(f"predicted: {pred_norm}, actual: {label}")

    # now write the data to a .bin file
    filename = os.path.join(DATA_CACHE_DIR, f"bb-cio_val.bin")
    write_evalfile(filename, datas)

if __name__ == "__main__":
    #with open("bb-cio/dataset.jsonl", "w") as file:
    #    for ex in iterate_examples():
    #        file.write(json.dumps(ex)+"\n")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_type", type=str, default="gpt2", help="the model type to use")
    parser.add_argument("-d", "--device", type=str, default="cuda", help="the device to use")
    parser.add_argument("-n", "--n_samples", type=int, default=-1, help="Max number of samples to evaluate")
    args = parser.parse_args()
    evaluate(args.model_type, args.device, args.n_samples)
