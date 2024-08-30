import argparse

from transformers import pipeline
from datasets import load_dataset
import torch
import pandas as pd
from thefuzz import fuzz

parser = argparse.ArgumentParser(description="ARBITER accuracy evaluation")
parser.add_argument("-m", "--model_path", type=str, default="jrahn/arbitersim_2m_3e_gpt2_124M_hf", help="Path to Hugging Face model")
parser.add_argument("-d", "--dataset", type=str, default="jrahn/arbiter_2m", help="Path to Hugging Face dataset")
parser.add_argument("-s", "--split", type=str, default="test", help="Dataset split to evaluate")
parser.add_argument("-g", "--greedy", action="store_true", help="Use greedy decoding")
parser.add_argument("-t", "--temp", type=float, default=0.6, help="Sampling temperature")
parser.add_argument("-k", "--topk", type=int, default=5, help="Sampling top-k")
parser.add_argument("-b", "--batch_size", type=int, default=1, help="Batch size")
args = parser.parse_args()

pipe = pipeline("text-generation", model=args.model_path, device_map="auto", torch_dtype=torch.bfloat16, batch_size=args.batch_size)
pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id
pipe.tokenizer.padding_side = "left"

def prepare(e):
    previous_state, action, recent_moves, new_state, reward, terminated, truncated = e["text"].split("+")
    e["prompt"] = f"{previous_state}+{action}+{recent_moves}+"
    e["target"] = f"{new_state}+{reward}+{terminated}+{truncated}"
    return e

ds = load_dataset(args.dataset, split=args.split)
# DEBUG: limit to 10 samples for faster evaluation
#ds = ds.select(range(10))  # limit to 10 samples for faster evaluation

# sample data format
# '8/p2Q2N1/4p3/r1P1P3/3p4/6NP/r4PP1/3K3R b - - 3 44+a5a4+f7e8 a6a5 e8c6 f3a3 c6b7 d5d4 b7d7 a3a2 d2d1 a5a4+8/p2Q2N1/4p3/2P1P3/r2p4/6NP/r4PP1/3K3R w - - 4 45+0.001+0+0'
# split on "+" into parts: previous_state, action, recent_moves, new_state, reward, terminated, truncated

ds = ds.map(prepare)

print(f"Evaluating ArbiterSim model on {args.split} split of {args.dataset} dataset ...")

gen_args = {
    "max_length": 256,
    "truncation": True,
    "pad_token_id": pipe.tokenizer.eos_token_id,
    "return_full_text": False,
    "batch_size": args.batch_size
}

if args.greedy:
    preds = [g[0]["generated_text"] for g in pipe(ds["prompt"], do_sample=False, **gen_args)]
else:
    preds = [g[0]["generated_text"] for g in pipe(ds["prompt"], do_sample=True, temperature=args.temp, top_k=args.topk, **gen_args)]


def evaluate_completion(target, prediction):
    new_state, reward, terminated, trucated = target.split("+")
    try:
        p_new_state, p_reward, p_terminated, p_truncated = prediction.split("+")
        invalid = False
    except ValueError:
        p_new_state, p_reward, p_terminated, p_truncated = None, None, None, None
        invalid = True
    try:
        reward_mae = abs(float(reward) - float(p_reward))
    except ValueError:
        reward_mae = None
    result = {
        "invalid": invalid, 
        "next_state_correct": new_state == p_new_state, 
        "next_state_fuzzratio": fuzz.ratio(new_state, p_new_state), 
        "reward": reward,
        "reward_correct": reward == p_reward, 
        "reward_mae": reward_mae, 
        "terminated": terminated,
        "terminated_correct": terminated == p_terminated, 
        "truncated": trucated,
        "truncated_correct": trucated == p_truncated
        }
    return result

results = []
for sample, pred in zip(ds, preds):
    results.append(evaluate_completion(sample["target"], pred))

with open("arbiter_accuracy_results.csv", "w") as f:
    df = pd.DataFrame(results)
    df.to_csv(f, index=False)

stats = {
    "total": len(results),
    "invalid_completion": sum(r["invalid"] for r in results),
    "next_state_correct": sum(r["next_state_correct"] for r in results),
    "next_state_fuzzratio": [r["next_state_fuzzratio"] for r in results if r["next_state_fuzzratio"] > 0],
    "reward_correct": sum(r["reward_correct"] for r in results),
    "reward_mae": [r["reward_mae"] for r in results if r["reward_mae"] is not None],
    "terminated_correct": sum(r["terminated_correct"] for r in results),
    "truncated_correct": sum(r["truncated_correct"] for r in results),
}

print("Evaluation results:")
print(f"Total samples: {stats['total']:,}")
print(f"Invalid completions: {stats['invalid_completion']/stats['total']:.2%}")
print(f"Next state correct: {stats['next_state_correct']/stats['total']:.2%}")
print(f"Next state fuzzratio: {sum(stats['next_state_fuzzratio'])/len(stats['next_state_fuzzratio'])/100:.2%}")
print(f"Reward correct: {stats['reward_correct']/stats['total']:.2%}")
print(f"Reward MAE: {sum(stats['reward_mae'])/len(stats['reward_mae']):.4f}")
print(f"Terminated correct: {stats['terminated_correct']/stats['total']:.2%}")
print(f"Truncated correct: {stats['truncated_correct']/stats['total']:.2%}")
