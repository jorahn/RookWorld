import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from vllm import LLM, SamplingParams
from collections import deque
from tqdm import tqdm

from joblib import Parallel, delayed

from argparse import ArgumentParser

parser = ArgumentParser(description="Run RookWorld self-play in vllm")
parser.add_argument("-m", "--model", type=str, help="Hugging Face RookWorld model")
parser.add_argument("-b", "--batch_size", type=int, default=64, help="Inference Batch Size = Parallel Games")
parser.add_argument("-l", "--log_file", type=str, help="Logfile to record rollouts")
parser.add_argument("-d", "--debug", action="store_true", help="Run in DEBUG")
args = parser.parse_args()

STARTING_POSITION = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
NUM_GPUS = 1 # increase only if model doesn't fit on one gpu, else run multiple processese with CUDA_VISIBLE_DEVICES in parallel
DEBUG = args.debug
BATCH_SIZE = args.batch_size
if args.log_file:
    raise NotImplementedError("logging rollouts is not implemented yet")

def process_action(text):
    try:
        action = text.split("B: ")[-1].strip()
    except IndexError:
        action = "0000"
    return action

def batch_process_actions(outputs, n_jobs=-1):
    return Parallel(n_jobs=n_jobs)(delayed(process_action)(o.outputs[0].text) for o in outputs)

def process_environment(text, action, history):
    try:
        next_state, reward, terminated, truncated = text.split("+")
        #reward = float(reward)
        terminated, truncated = int(terminated), int(truncated)
        if terminated or truncated or action == "0000":
            game = {"state": STARTING_POSITION, "history": deque(maxlen=10), "terminated": terminated, "truncated": truncated}
        else:
            game = {"state": next_state, "history": history, "terminated": 0, "truncated": 0}
    except (ValueError, TypeError):
        game = {"state": STARTING_POSITION, "history": deque(maxlen=10), "terminated": 0, "truncated": 1}
    return game

def batch_process_environment(inputs, n_jobs=-1):
    return Parallel(n_jobs=n_jobs)(delayed(process_environment)(output.outputs[0].text, action, history) for output, action, history in inputs)

def make_policy_prompt(state):
    return "P: "+state+" "

def batch_make_policy_prompts(games, n_jobs=-1):
    return Parallel(n_jobs=n_jobs)(delayed(make_policy_prompt)(game["state"]) for game in games)


def make_environment_prompt(state, action, history):
    history_text = " ".join(history)
    return f"A: {state}+{action}+{history_text}+"

def batch_make_environment_prompts(inputs, n_jobs=-1):
    return Parallel(n_jobs=n_jobs)(delayed(make_environment_prompt)(state, action, history) for state, action, history in inputs)


games = []
for i in range(BATCH_SIZE):
    games.append({"state": STARTING_POSITION, "history": deque(maxlen=10)})
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=144)
llm = LLM(
    model="rookworld_7m_3e_gpt2_124M_hf", 
    tokenizer="rookworld_7m_3e_gpt2_124M_hf", 
    tensor_parallel_size=NUM_GPUS,
)

pbar_actions = tqdm(desc="Actions")
pbar_episodes = tqdm(desc="Episodes")
while True:
    # state -> policy
    prompts = batch_make_policy_prompts(games)
    histories = [g["history"] for g in games]
    if DEBUG: print("\n", prompts)
    outputs = llm.generate(prompts, sampling_params)
    if DEBUG: print("\n", outputs)
    actions = batch_process_actions(outputs)
    if DEBUG: print("\n", actions)
    for i, a in enumerate(actions):
        histories[i].append(a)
    pbar_actions.update(len(actions))

    # action -> environment
    states = [game["state"] for game in games]
    inputs = zip(states, actions, histories)
    prompts = batch_make_environment_prompts(inputs)
    if DEBUG: print("\n", prompts)
    outputs = llm.generate(prompts, sampling_params)
    inputs = zip(outputs, actions, histories)
    games = batch_process_environment(inputs)
    if DEBUG: print("\n", games)
    pbar_episodes.update(sum(g["terminated"] for g in games))
    if DEBUG: input("press ENTER to continue...")
