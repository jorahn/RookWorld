# self-play of RookWorld policy in RookWorld environment
# allow batched generation of both policy and environment

# add flag to log the policies, environment, or both
# add flag to only log the actions taken by the winning policy
# TODO: add flag to use different models for the two player policies and the environment

import argparse
from collections import deque
import os

# suppress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from transformers import pipeline
import transformers
import torch
from tqdm import tqdm

# suppress the warning about using the pipeline sequentially on GPU
transformers.logging.set_verbosity_error()

args = argparse.ArgumentParser(description="RookWorld self-play rollout generation")
args.add_argument("-p", "--policy", type=str, default="jrahn/rookworld_7m_3e_gpt2_124M_hf", help="Policy model")
args.add_argument("-p2", "--policy_player2", type=str, default=None, help="Player 2 model, if different from player 1")
args.add_argument("-e", "--env", type=str, default=None, help="Environment model, if different from policy")
args.add_argument("-l", "--log", type=str, default="both", choices=["policy", "env", "both"], help="Log generations from policy|env|both")
args.add_argument("-w", "--winning_policy", action="store_true", help="Skip logging generations from losing or drawing policy")
args.add_argument("-o", "--output", type=str, default="rookworld_selfplay.txt", help="Log file")
args.add_argument("-n", "--num_rollouts", type=int, default=2, help="Number of rollouts to generate")
args.add_argument("-bs", "--batch_size", type=int, default=2, help="Batch size for generation")
args = args.parse_args()

assert not(args.winning_policy and args.log == "env"), "Cannot log only the winning policy when logging only the environment"

if args.policy_player2 is not None:
    raise NotImplementedError("Different models for player 1 and player 2 are not yet supported")
if args.env is not None:
    raise NotImplementedError("Different models for the environment are not yet supported")

STARTING_POSITION = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
device_map = "auto"

# load the model
pipe = pipeline(
    "text-generation", 
    model=args.policy,
    device_map=device_map, 
    torch_dtype=torch.bfloat16, 
    batch_size=args.batch_size
)

pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id
pipe.tokenizer.padding_side = "left"
gen_args = {
    "do_sample": True,
    "temperature": 0.7,
    "top_k": 15,
    "max_length": 256,
    "truncation": True, 
    "pad_token_id": pipe.tokenizer.eos_token_id,
    "return_full_text": False,
}

current_games = [(STARTING_POSITION, "0000", deque(maxlen=10)) for _ in range(args.batch_size)]
logs = [[] for _ in range(args.batch_size)]
pbar_rollouts = tqdm(total=args.num_rollouts, desc="Rollouts", position=0)
pbar_pactions = tqdm(desc="Policy Actions", position=1)

# generate rollouts
while True:
    # generate the next move
    policy_prompts = [f"P: {state} " for state, _, _ in current_games]
    policy_generations = pipe(policy_prompts, **gen_args)

    # extract action, update current_games and logs
    for i, g in enumerate(policy_generations):
        result = g[0]["generated_text"]
        # TODO verify that the generated text is a valid policy response
        
        if args.log in ("policy", "both"):
            logs[i].append(policy_prompts[i] + result)
        try:
            # validate the policy response
            action = result.rsplit("B: ")[1].strip()
            state, _, history = current_games[i]
            history.append(action)
            current_games[i] = (state, action, history)
            pbar_pactions.update(1)
        except IndexError as e:
            # invalid action, skip this generation, reset the game
            current_games[i] = [(STARTING_POSITION, "0000", deque(maxlen=10))]
            logs[i] = []
    
    # generate the environment response
    env_prompts = [f"A: {state}+{action}+{' '.join(history)}+" for state, action, history in current_games]
    env_generations = pipe(env_prompts, **gen_args)

    # extract state, update current_games and logs
    for i, g in enumerate(env_generations):
        result = g[0]["generated_text"]
        # TODO verify that the generated text is a valid environment response
        if args.log in ("env", "both"):
            logs[i].append(env_prompts[i] + result)
        try:
            # validate the environment response
            new_state, reward, terminated, truncated = result.split("+")
            reward = float(reward)
            terminated, truncated = int(terminated), int(truncated)
        except (ValueError, TypeError) as e:
            # invalid response, skip this generation, reset the game
            current_games[i] = (STARTING_POSITION, "0000", deque(maxlen=10))
            logs[i] = []
            continue
        
        # check if invalid policy response
        if current_games[i][1] == "0000":
            # skip all logs from the rollout, reset the game
            current_games[i] = (STARTING_POSITION, "0000", deque(maxlen=10))
            logs[i] = []

        elif terminated or truncated:
            # log the game
            if logs[i]:
                last_state, _, _ = current_games[i]
                last_player = last_state.split(" ")[1] # player w|b in state FEN

                with open(f"{args.output}", "a") as f:
                    if args.winning_policy and abs(reward) == 1.0:
                        # only log the generations of by the winning policy
                        # and log the generations of the environment, if requested
                        for row in logs[i]:
                            if row.startswith("A: "): # environment action
                                f.write(row+"\n")
                            else: # policy action
                                # only log actions taken by last_player
                                policy_player = row.split(" ")[2] # player w|b in prefix + state FEN + ...
                                if (reward == 1.0 and policy_player == last_player) or (reward == -1.0 and policy_player != last_player):
                                    f.write(row+"\n")
                        with open("debug-log.txt", "a") as debug_f:
                            debug_f.write("\n".join(logs[i]) + "\n")
                    elif args.winning_policy:
                        # don't log draws
                        pass
                    else:
                        f.write("\n".join(logs[i]) + "\n")
            else:
                # nothing to log
                logs[i] = []

            # complete the rollout, reset the game
            if (args.winning_policy and abs(reward) == 1.0) or not args.winning_policy:
                # discard draws
                pbar_rollouts.update(1)
            current_games[i] = (STARTING_POSITION, "0000", deque(maxlen=10))
            logs[i] = []

        else:
            # if the game is not over, update the state
            current_games[i] = (new_state, current_games[i][1], current_games[i][2])

    if pbar_rollouts.n >= args.num_rollouts:
        pbar_rollouts.close()
        pbar_pactions.close()
        break

print("Done")