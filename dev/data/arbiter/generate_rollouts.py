# This script generates rollouts of chess games using python-chess as the environment and a ROOK model as policy.

# improvements:
# - generate more truncated and terminated episodes
# - generate sufficient examples for rate states like
#   - checkmate
#   - stalemate
#   - insufficient material
#   - 50/75-move rule
#   - 3/5-fold repetition
#   - piece promotion, en-passant, pinned pieces, king in check

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
args.add_argument("-pm", "--policy_model", type=str, default="jrahn/rook_5m_3e_gpt2_124M_hf", help="Model name or path")
args.add_argument("-em", "--env_model", type=str, default="", help="Use ArbiterSim or RookWorld instead of ChessEnvironment")
args.add_argument("-bs", "--batch_size", type=int, default=64, help="Batch size of concurrent games (64 ~ 6GB VRAM)")
args.add_argument("-t", "--temperature", type=float, default=0.7, help="Sampling temperature")
args.add_argument("-k", "--top_k", type=int, default=15, help="Top-K sampling")
args.add_argument("-g", "--greedy", action="store_true", default=False, help="Use greedy sampling (not recommended, lacks diversity)")
args.add_argument("-l", "--logfile", type=str, default="chess_env_rollouts.jsonl", help="Save rollouts to this file")
args.add_argument("-le", "--log_evol", action="store_true", default=False, help="Log Policy & Environment, select only moves from winning side for RookWorld Evol")
args = args.parse_args()

if args.log_evol:
    raise NotImplementedError("log_evol is not implemented yet")

STARTING_POSITION = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

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

class RookWorldEnvironment:
    def __init__(self, model, batch_size=1):
        self.current_position = [STARTING_POSITION] * batch_size
        self.previous_moves = [deque(maxlen=10)] * batch_size
        self.m = pipeline("text-generation", model=model, device_map="auto", 
                          torch_dtype=torch.bfloat16, batch_size=batch_size)
        self.m.tokenizer.pad_token_id = self.m.model.config.eos_token_id
        self.m.tokenizer.padding_side = "left" 
        self.batch_size = batch_size

    def step(self, actions):
        states = self.current_position
        results = [[]] * self.batch_size

        # TODO maybe move to end of step in sync with ChessEnvironment
        for i, action in enumerate(actions):
            if action is not None:
                self.previous_moves[i].append(action)
            else:
                self.previous_moves[i].append("None")
        moves = [" ".join(pm) for pm in self.previous_moves]

        sam = zip(states, actions, moves)
        prompts = [f"A: {s}+{a}+{m}+" for s, a, m in sam]

        # TODO implement sampling vs greedy decoding like in Policy
        gens = self.m(prompts, max_length=256, truncation=True, pad_token_id=self.m.tokenizer.eos_token_id, return_full_text=False)

        txts = [gen[0]['generated_text'] for gen in gens]
        #print(txt)
        for i, txt in enumerate(txts):
            try:
                assert actions[i] is not None
                new_state, reward, terminated, truncated = txt.split("+")
                reward = float(reward)
                terminated = int(terminated)
                truncated = int(truncated)
            except ValueError:
                new_state = None
                reward = 0.0
                terminated = True
                truncated = True
            except AssertionError:
                current_player = states[i].split(" ")[1]
                assert current_player in ["w", "b"], f"Invalid state FEN generated: {states[i]}"
                new_state = None
                reward = -1.0 if current_player == "w" else 1.0
                terminated = False
                truncated = True
                action = "None"
            info = {
                "previous_state": self.current_position[i],
                "action": action,
                "new_state": new_state,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "recent_moves": moves[i],
            }
            results[i] = (new_state, reward, terminated, truncated, info)
        

        self.current_position = [new_state for new_state, _, _, _, _ in results]
        return results

    def reset(self, n=None, fen=None, recent_moves=[]):
        if n:
            if fen is not None:
                self.current_position[n] = fen
                self.recent_moves[n] = deque(recent_moves, maxlen=10)
            else:
                self.current_position[n] = STARTING_POSITION
                self.recent_moves[n] = deque(maxlen=10)
            return self.current_position[n], None
        else:
            if fen is not None:
                self.current_position = [fen] * self.batch_size
                self.recent_moves = [deque(recent_moves, maxlen=10)] * self.batch_size
            else:
                self.current_position = [STARTING_POSITION] * self.batch_size
                self.recent_moves = [deque(maxlen=10)] * self.batch_size
            return list(zip(self.current_position, [None] * self.batch_size))

class ArbiterSimEnvironment:
    # Use ArbiterSim model to simulate chess game environment
    # TODO allow batched inference like Policy

    def __init__(self, model):
        self.current_position = STARTING_POSITION
        self.previous_moves = deque(maxlen=10)
        self.m = pipeline("text-generation", model=model, device_map="auto", 
                          torch_dtype=torch.bfloat16, batch_size=1)
        self.m.tokenizer.pad_token_id = self.m.model.config.eos_token_id
        self.m.tokenizer.padding_side = "left"
    
    def step(self, action):
        state = self.current_position
        
        # TODO maybe move to end of step in sync with ChessEnvironment
        if not action:
            current_player = state.split(" ")[1]
            assert current_player in ["w", "b"], f"Invalid state FEN generated: {state}"
            new_state = None
            reward = -1.0 if current_player == "w" else 1.0
            terminated = False
            truncated = True
        else:
            self.previous_moves.append(action)
            moves = " ".join(self.previous_moves)
            
            if "rookworld" in args.env_model.lower():
                prompt = f"A: {state}+{action}+{moves}+"
            else:
                prompt = f"{state}+{action}+{moves}+"

            # TODO implement sampling vs greedy decoding like in Policy
            gen = self.m(prompt, max_length=256, truncation=True, pad_token_id=self.m.tokenizer.eos_token_id, return_full_text=False)

            txt = gen[0]['generated_text']
            #print(txt)
            try:
                new_state, reward, terminated, truncated = txt.split("+")
                reward = float(reward)
                terminated = int(terminated)
                truncated = int(truncated)
            except ValueError:
                new_state = None
                reward = 0.0
                terminated = True
                truncated = True

        info = {
            "previous_state": state,
            "action": action,
            "new_state": new_state,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "recent_moves": list(self.recent_moves),
        }
        self.current_position = new_state
        return new_state, reward, terminated, truncated, info

    def reset(self, fen=None, recent_moves=[]):
        if fen is not None:
            self.current_position = fen
            self.recent_moves = deque(recent_moves, maxlen=10)
        else:
            self.current_position = STARTING_POSITION
            self.recent_moves = deque(maxlen=10)
        return self.current_position, None
    
class ChessEnvironment:
    # try to mirror the API of OpenAI Gym
    # TODO maybe subclass it

    def __init__(self, legal_move_reward=0.001):
        self.board = chess.Board()
        self.illegal_move = False
        self.recent_moves = deque(maxlen=10) # to detect 5-fold repetition
        # maybe make more efficient, last move is already known and 9 half-moves are enough

        self.legal_move_reward = legal_move_reward
    
    def get_state(self):
        return self.board.fen()
    
    def action_space(self):
        return self.board.legal_moves
    
    def step(self, action):
        previous_state = self.get_state()

        # TODO move to end of step, current action is already known
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

    def reset(self, fen=None):
        self.board = chess.Board()
        if fen is not None:
            self.board.set_fen(fen)
        self.illegal_move = False
        self.recent_moves.clear()
        info = None
        return self.get_state(), info
    

def play(n_episodes, model, envs, batch_size=2, sampling={}):
    # TODO maybe add max_half_moves limit

    # TODO potentially implement two alternating agents, to allow for self-play of different models
    agent = Policy(model=model, batch_size=batch_size, sampling=sampling)
    
    stats = {"terminated": 0, "truncated": 0, "episodes": 0, "steps": 0}
    with open(args.logfile, "a") as log:
        pbar_episodes = tqdm(range(n_episodes), desc="Episodes", position=0)

        if "rookworld" in args.env_model.lower():
            state_info_tuples = envs.reset()
        else:
            state_info_tuples = [e.reset() for e in envs]
        states = [s for s, _ in state_info_tuples]
        pbar_steps = tqdm(desc="Steps", position=1)
    
        done = False
        # loop over episodes
        while not done:
            actions = agent.sample_actions(states)
            if "rookworld" in args.env_model.lower():
                observations = envs.step(actions)
            else:
                observations = [e.step(a) for e, a in zip(envs, actions)]
            states = []
            for i, (state, reward, termination, truncation, info) in enumerate(observations):
                states.append(state)
                pbar_steps.update(1)
                stats["steps"] += 1

                # TODO maybe enable both players to get a reward
                # not required for Env but for RL in Env

                if info is not None:
                    # TODO log only winning games for RookWorld Evol
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
                    if "rookworld" in args.env_model.lower():
                        state_info_tuples[i] = envs.reset(i)
                    else:
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
    
    if args.env_model:
        if "rookworld" in args.env_model.lower():
            envs = RookWorldEnvironment(model=args.env_model, batch_size=args.batch_size)
            env_name = "RookWorldEnvironment"
        else:
            envs = [ArbiterSimEnvironment(model=args.env_model) for _ in range(args.batch_size)]
            env_name = "ArbiterSimEnvironment"
    else:
        envs = [ChessEnvironment() for _ in range(args.batch_size)]
        env_name = "ChessEnvironment"

    print(f"Playing games (Environment: {env_name})...")
    stats = play(
        n_episodes=args.episodes, 
        model=args.policy_model, 
        envs=envs,
        batch_size=args.batch_size,
        sampling=sampling,
    )

    print("Done!")
    print("-" * 30)
    print("Episodes:", stats["episodes"])
    print("Truncated Episodes:", stats["truncated"], f"- ({stats['truncated'] / stats['episodes']:.2%})")
    print("Terminated Episodes:", stats["terminated"], f"- ({stats['terminated'] / stats['episodes']:.2%})")
    print("Total Steps:", stats["steps"])
    print("-" * 30)
    print(f"Check {args.logfile} for the rollout data.")
