# Market Making with DQN from LOB

This project recreates the core logic of the paper:
**"Market Making with Deep Reinforcement Learning from Limit Order Books"**.

It includes:
- LOB state with rolling window
- dynamic market features (OSI, RV, RSI)
- agent state (inventory, remaining time)
- Attn-LOB encoder (cnn + inception-style temporal conv + self-attention)
- continuous quote design `(A1, A2)` converted to executable bid/ask prices
- hybrid reward `DP + TP - IP`
- DQN training loop with replay buffer and target network

## Project structure

- `mm_dqn/config.py` - dataclasses and hyperparameters
- `mm_dqn/features.py` - LOB normalization and dynamic features
- `mm_dqn/model.py` - Attn-LOB and DQN head
- `mm_dqn/env.py` - event-driven market making simulator
- `mm_dqn/replay.py` - replay buffer
- `mm_dqn/agent.py` - DQN agent
- `train.py` - end-to-end training script

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python train.py
```

## Notes

- This implementation is paper-faithful in modeling choices, but uses synthetic event data by default.
- TODO: Replace synthetic generator with real LOB events to run realistic backtests.
