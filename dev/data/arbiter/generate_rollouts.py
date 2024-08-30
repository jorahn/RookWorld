# This script generates rollouts of chess games using python-chess as the environment and a ROOK model as policy.

from argparse import ArgumentParser
from collections import deque
import json

from transformers import pipeline
import torch
import chess
from tqdm import tqdm

from transformers.utils import logging
logging.get_logger("transformers").setLevel(logging.ERROR)

args = ArgumentParser()
args.add_argument("-e", "--episodes", type=int, default=100, help="Number of episodes to play")
args.add_argument("-m", "--model", type=str, default="jrahn/rook_5m_3e_gpt2_124M_hf", help="Model name or path")
args.add_argument("-bs", "--batch_size", type=int, default=64, help="Batch size of concurrent games (64 ~ 6GB VRAM)")
args.add_argument("-t", "--temperature", type=float, default=0.7, help="Sampling temperature")
args.add_argument("-k", "--top_k", type=int, default=15, help="Top-K sampling")
args.add_argument("-g", "--greedy", action="store_true", default=False, help="Use greedy sampling (not recommended, lacks diversity)")
args.add_argument("-l", "--logfile", type=str, default="chess_env_rollouts.jsonl", help="Save rollouts to this file")
args = args.parse_args()

class Policy:
    # to learn various states, including illegal actions, it's helpful to have an imperfect policy
    # maybe random rollouts would be sufficient, but a model can be used to generate more diverse states
    def __init__(self, model="jrahn/rook_5m_3e_gpt2_124M_hf", device_map="auto", sampling={}, batch_size=1):
        self.m = pipeline("text-generation", model=model, device_map=device_map, 
                          torch_dtype=torch.bfloat16, batch_size=batch_size)
        self.m.tokenizer.pad_token_id = self.m.model.config.eos_token_id
        self.m.tokenizer.padding_side = "left"
        sampling_defaults = {
            "do_sample": True,
            "temperature": 0.7,
            "top_k": 15,
        }
        self.sampling = "greedy" if sampling == "greedy" else {**sampling_defaults, **sampling}

    def sample_actions(self, states):
        prompts = ["P: " + s for s in states] # prefix
        if self.sampling == "greedy":
            gens = self.m(prompts, do_sample=False, max_length=256, truncation=True,
                         pad_token_id=self.m.tokenizer.eos_token_id)
        else:
            gens = self.m(prompts, **self.sampling, max_length=256, truncation=True,
                         pad_token_id=self.m.tokenizer.eos_token_id)

        txts = [g[0]['generated_text'] for g in gens]
        moves = []
        for txt in txts:
            try:
                move = txt.rsplit("B: ", 1)[1].strip()
                assert len(move) in [4, 5]
            except (IndexError, AssertionError):
                move = None
            moves.append(move)
        return moves

class ChessEnvironment:
    # try to mirror the API of OpenAI Gym
    # TODO maybe subclass it

    def __init__(self, legal_move_reward=0.001, log_file=None):
        self.board = chess.Board()
        self.log_file = log_file
        self.illegal_move = False
        self.recent_moves = deque(maxlen=10) # to detect 5-fold repetition
        self.legal_move_reward = legal_move_reward
    
    def get_state(self):
        return self.board.fen()
    
    def action_space(self):
        return self.board.legal_moves
    
    def step(self, action):
        previous_state = self.get_state()

        self.recent_moves.append(action)
        try:
            move = chess.Move.from_uci(action)
            self.board.push(move)
        except (ValueError, AssertionError, TypeError): # (chess.InvalidMoveError, chess.IllegalMoveError)
            self.illegal_move = True
        
        outcome = self.board.outcome()
        if outcome is not None:
            if outcome.winner is not None:
                # the reward is associated with the player who made the last move
                # board.turn is the player who will make the next move
                reward = 1.0 if outcome.winner != self.board.turn else -1.0 
            else:
                reward = 0.5
        else:
            reward = self.legal_move_reward
        reward = reward if not self.illegal_move else -1.0

        observation = self.get_state()
        terminated = self.board.is_game_over()
        truncated = self.illegal_move # True if illegal move or flagging
        info = {
            "previous_state": previous_state,
            "action": action,
            "new_state": observation,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "recent_moves": list(self.recent_moves),
            "_illegal_move": self.illegal_move,
            "_outcome": str(outcome),
            "_legal_move_reward": self.legal_move_reward,
        }
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, fen=None):
        # seed just for Gym compatibility, could be used for Chess960
        self.board = chess.Board()
        if fen is not None:
            self.board.set_fen(fen)
        self.illegal_move = False
        self.recent_moves.clear()
        info = None
        return self.get_state(), info
    

def play(n_episodes, model, batch_size=2, sampling={}):
    # TODO maybe add max_half_moves limit

    envs = [ChessEnvironment() for _ in range(batch_size)]

    # TODO potentially implement two alternating agents, to allow for self-play of different models
    agent = Policy(model=model, batch_size=batch_size, sampling=sampling)
    
    stats = {"terminated": 0, "truncated": 0, "episodes": 0, "steps": 0}
    with open(args.logfile, "a") as log:
        pbar_episodes = tqdm(range(n_episodes), desc="Episodes", position=0)

        state_info_tuples = [e.reset() for e in envs]
        states = [s for s, _ in state_info_tuples]
        pbar_steps = tqdm(desc="Steps", position=1)
    
        done = False
        # loop over episodes
        while not done:
            actions = agent.sample_actions(states)
            observations = [e.step(a) for e, a in zip(envs, actions)]
            states = []
            for i, (state, reward, termination, truncation, info) in enumerate(observations):
                states.append(state)
                pbar_steps.update(1)
                stats["steps"] += 1

                # TODO maybe enable both players to get a reward
                # not required for Env but for RL in Env

                if info is not None:
                    log.write(json.dumps(info) + "\n")
                
                if termination or truncation: # maybe also check for max_half_moves
                    stats["terminated"] += termination
                    stats["truncated"] += truncation
                    pbar_episodes.update(1)
                    stats["episodes"] += 1
                    if pbar_episodes.n == n_episodes:
                        done = True
                        # log will contain n_episodes finished
                        # + up to (batch_size - 1) started, unfinished episodes
                        # TODO maybe track num_started_episodes and ignore in logging once > n_episodes
                    state_info_tuples[i] = envs[i].reset()
                    states[i] = state_info_tuples[i][0]
    
    pbar_episodes.close()
    pbar_steps.close()
    return stats
            
if __name__ == "__main__":
    if args.greedy:
        sampling = "greedy"
    else:
        sampling = {
            "temperature": args.temperature,
            "top_k": args.top_k,
        }

    print("Playing games...")
    stats = play(
        n_episodes=args.episodes, 
        model=args.model, 
        batch_size=args.batch_size,
        sampling=sampling
    )

    print("Done!")
    print("-" * 30)
    print("Episodes:", stats["episodes"])
    print("Truncated Episodes:", stats["truncated"], f"- ({stats['truncated'] / stats['episodes']:.2%})")
    print("Terminated Episodes:", stats["terminated"], f"- ({stats['terminated'] / stats['episodes']:.2%})")
    print("Total Steps:", stats["steps"])
    print("-" * 30)
    print(f"Check {args.logfile} for the rollout data.")
