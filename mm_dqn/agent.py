from __future__ import annotations

import random
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

from mm_dqn.config import EnvConfig, ModelConfig, TrainConfig
from mm_dqn.model import DQNNet
from mm_dqn.replay import ReplayBuffer, Transition


class DQNAgent:
    def __init__(self, env_cfg: EnvConfig, model_cfg: ModelConfig, train_cfg: TrainConfig, device: str = "cpu"):
        self.env_cfg = env_cfg
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.device = device
        self.steps = 0

        self.q = DQNNet(
            n_levels=env_cfg.n_levels,
            dynamic_dim=model_cfg.dynamic_dim,
            agent_dim=model_cfg.agent_dim,
            hidden_dim=model_cfg.hidden_dim,
            action_dim=model_cfg.action_dim,
            n_heads=model_cfg.n_heads,
        ).to(device)
        self.target_q = DQNNet(
            n_levels=env_cfg.n_levels,
            dynamic_dim=model_cfg.dynamic_dim,
            agent_dim=model_cfg.agent_dim,
            hidden_dim=model_cfg.hidden_dim,
            action_dim=model_cfg.action_dim,
            n_heads=model_cfg.n_heads,
        ).to(device)
        self.target_q.load_state_dict(self.q.state_dict())
        self.opt = optim.Adam(self.q.parameters(), lr=train_cfg.lr)
        self.buffer = ReplayBuffer(train_cfg.buffer_size)

    def epsilon(self) -> float:
        p = min(1.0, self.steps / max(1, self.train_cfg.epsilon_decay_steps))
        return self.train_cfg.epsilon_start + p * (self.train_cfg.epsilon_end - self.train_cfg.epsilon_start)

    @torch.no_grad()
    def act(self, lob: np.ndarray, dyn: np.ndarray, agent: np.ndarray) -> int:
        if random.random() < self.epsilon():
            return random.randrange(self.model_cfg.action_dim)
        return self.act_greedy(lob, dyn, agent)

    @torch.no_grad()
    def act_greedy(self, lob: np.ndarray, dyn: np.ndarray, agent: np.ndarray) -> int:
        lob_t = torch.from_numpy(lob).unsqueeze(0).to(self.device)
        dyn_t = torch.from_numpy(dyn).unsqueeze(0).to(self.device)
        ag_t = torch.from_numpy(agent).unsqueeze(0).to(self.device)
        qv = self.q(lob_t, dyn_t, ag_t)
        return int(torch.argmax(qv, dim=-1).item())

    def push(self, t: Transition) -> None:
        self.buffer.push(t)

    def train_step(self) -> Tuple[float, float, bool]:
        if len(self.buffer) < max(self.train_cfg.batch_size, self.train_cfg.warmup_steps):
            return 0.0, 0.0, False
        batch = self.buffer.sample(self.train_cfg.batch_size)

        lob = torch.from_numpy(np.stack([x.lob for x in batch])).to(self.device)
        dyn = torch.from_numpy(np.stack([x.dyn for x in batch])).to(self.device)
        ag = torch.from_numpy(np.stack([x.agent for x in batch])).to(self.device)
        act = torch.from_numpy(np.asarray([x.action for x in batch], dtype=np.int64)).to(self.device)
        rew = torch.from_numpy(np.asarray([x.reward for x in batch], dtype=np.float32)).to(self.device)
        n_lob = torch.from_numpy(np.stack([x.next_lob for x in batch])).to(self.device)
        n_dyn = torch.from_numpy(np.stack([x.next_dyn for x in batch])).to(self.device)
        n_ag = torch.from_numpy(np.stack([x.next_agent for x in batch])).to(self.device)
        done = torch.from_numpy(np.asarray([x.done for x in batch], dtype=np.float32)).to(self.device)

        q = self.q(lob, dyn, ag).gather(1, act.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_action = torch.argmax(self.q(n_lob, n_dyn, n_ag), dim=-1)
            target_q = self.target_q(n_lob, n_dyn, n_ag).gather(1, next_action.unsqueeze(1)).squeeze(1)
            y = rew + (1.0 - done) * self.train_cfg.gamma * target_q
        loss = F.smooth_l1_loss(q, y)

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), max_norm=5.0)
        self.opt.step()

        if self.steps % self.train_cfg.target_update_steps == 0:
            self.target_q.load_state_dict(self.q.state_dict())
        return float(loss.item()), float(q.mean().item()), True
