# ArbiterSim

It is intended to be used for data collection for training a simulated chess environment (ArbiterSim) from python-chess as teacher. Given a state and a move, the model predicts the next state, termination and reward.  
  
The dynamics of a chess environment are deterministic, fully observable and straightforward to program - so it's not necessary to use a model to simulate it. The same can be said about the environments learned in the World Models paper by Ha and Schmidhuber. However, it can serve as a demonstration of how to learn an environment, use it as a simulation and train a policy on it.

Usage:
1. run `python generate_rollouts.py`
2. run `python make_dataset.py`
3. train Arbiter model using `scripts/run_gpt2_124M_arbiter_2m_3e.sh`
4. WIP use ArbiterSim as env
5. WIP train ROOK on self-play games played in ArbiterSim


### learn a chess world-model from python-chess

inputs:
- state (FEN), action (UCI), previous moves
  - arbiter needs to determine 3-fold/5-fold repetition and 50/75-moves
    - for repetitions, arbiter needs to keep a limited move stack record
    - the 50/75-moves counter is included in the state (FEN)
- initially ignore clock/flagging

return:
- new state (FEN)
- reward (small positive for legal move (0.001), 1 for win, -1 for loss/illegal, 0.5 for draw)
  - maybe revisit this for potential RL training
- termination (game ended with WLD)
- truncation (game ended with illegal move)


dataset:
- use python-chess to generate state, action, new-state + result
- create dataset from 
  - ROOK 5Mx3e model selfplay
    - implemented batched execution -> 48G VRAM: 1024 games in parallel
      - gameplay performance in batches was worse for ROOK evals, but not critical for this data generation
    - efficient packing: when one game in a batch is completed, that environment is reset and play continues immediately
    - regularly generates termination (full game) and truncation (illegal move) episodes -> good for learning the environment


approach:
- [x] lets implement batch selfplay (multi-gpu) to speed up data generation
- [x] implement data capture
- [x] generate dataset and push to HF
- [x] train ArbiterSim generator with llm.c
- [ ] implement ArbiterSim as selfplay environment
- [ ] selfplay in ArbiterSim -> collect data for supervised training or train with RL