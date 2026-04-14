from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque, namedtuple
import random


Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    """
    Experience replay buffer for DQN
    """
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        transitions = random.sample(self.buffer, batch_size)
        batch = Transition(*zip(*transitions))
        return batch

    def __len__(self):
        return len(self.buffer)


class DoubleDQNAgent:
    """
    NEW: Double DQN Agent
    - Uses online network for action selection
    - Uses target network for value evaluation
    - Reduces overestimation bias in Q-values
    """
    def __init__(
        self,
        model: nn.Module,
        target_model: nn.Module,
        lr: float = 1e-4,
        gamma: float = 0.99,
        device: str = 'cpu',
        use_double_dqn: bool = True
    ):
        self.model = model.to(device)
        self.target_model = target_model.to(device)
        self.device = device
        self.gamma = gamma
        self.use_double_dqn = use_double_dqn

        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Sync target network
        self.update_target()

    def update_target(self):
        """Copy weights from online network to target network"""
        self.target_model.load_state_dict(self.model.state_dict())

    def soft_update_target(self, tau: float = 0.005):
        """Soft update: target = tau * online + (1-tau) * target"""
        for target_param, online_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(tau * online_param.data + (1 - tau) * target_param.data)

    def select_action(self, state: dict, epsilon: float = 0.0) -> int:
        """
        Epsilon-greedy action selection
        state: dict with keys 'lob_window', 'dynamic_state', 'agent_state'
        """
        # Exploration: random action
        if random.random() < epsilon:
            return random.randint(0, self.model.mlp[-1].out_features - 1)

        # Exploitation: use model
        self.model.eval()
        with torch.no_grad():
            lob_window = torch.FloatTensor(state['lob_window']).unsqueeze(0).to(self.device)
            dynamic_state = torch.FloatTensor(state['dynamic_state']).unsqueeze(0).to(self.device)
            agent_state = torch.FloatTensor(state['agent_state']).unsqueeze(0).to(self.device)

            q_values = self.model(lob_window, dynamic_state, agent_state)
            action = q_values.argmax(dim=1).item()

        self.model.train()
        return action

    def train_step(self, batch, batch_size: int) -> dict:
        """
        Train the model on a batch of experiences
        Returns: dict with 'loss' and 'q_values' statistics
        """
        # Unpack batch
        lob_window = torch.FloatTensor(np.stack(batch.state[:, 0])).to(self.device)
        dynamic_state = torch.FloatTensor(np.stack(batch.state[:, 1])).to(self.device)
        agent_state = torch.FloatTensor(np.stack(batch.state[:, 2])).to(self.device)
        actions = torch.LongTensor(batch.action).to(self.device)
        rewards = torch.FloatTensor(batch.reward).to(self.device)
        next_lob_window = torch.FloatTensor(np.stack(batch.next_state[:, 0])).to(self.device)
        next_dynamic_state = torch.FloatTensor(np.stack(batch.next_state[:, 1])).to(self.device)
        next_agent_state = torch.FloatTensor(np.stack(batch.next_state[:, 2])).to(self.device)
        dones = torch.FloatTensor(batch.done).to(self.device)

        # Current Q values
        current_q_values = self.model(lob_window, dynamic_state, agent_state)
        q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q values
        if self.use_double_dqn:
            # Double DQN: use online network to select actions, target network to evaluate
            next_q_values_online = self.model(next_lob_window, next_dynamic_state, next_agent_state)
            next_actions = next_q_values_online.argmax(dim=1, keepdim=True)

            next_q_values_target = self.target_model(next_lob_window, next_dynamic_state, next_agent_state)
            next_q_values = next_q_values_target.gather(1, next_actions).squeeze(1)
        else:
            # Standard DQN
            next_q_values = self.target_model(next_lob_window, next_dynamic_state, next_agent_state)
            next_q_values = next_q_values.max(dim=1)[0]

        # TD target
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # Compute loss (Huber loss is more robust than MSE)
        loss = F.smooth_l1_loss(q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Statistics
        with torch.no_grad():
            stats = {
                'loss': loss.item(),
                'q_mean': q_values.mean().item(),
                'q_max': q_values.max().item(),
                'q_min': q_values.min().item(),
                'target_mean': target_q_values.mean().item(),
                'td_error': (target_q_values - q_values).abs().mean().item()
            }

        return stats


class OriginalDQNAgent:
    """
    Original DQN agent for comparison
    """
    def __init__(self, model: nn.Module, target_model: nn.Module, lr: float = 1e-4, gamma: float = 0.99, device: str = 'cpu'):
        self.model = model.to(device)
        self.target_model = target_model.to(device)
        self.device = device
        self.gamma = gamma

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.update_target()

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def select_action(self, state: dict, epsilon: float = 0.0) -> int:
        if random.random() < epsilon:
            return random.randint(0, self.model.mlp[-1].out_features - 1)

        self.model.eval()
        with torch.no_grad():
            lob_window = torch.FloatTensor(state['lob_window']).unsqueeze(0).to(self.device)
            dynamic_state = torch.FloatTensor(state['dynamic_state']).unsqueeze(0).to(self.device)
            agent_state = torch.FloatTensor(state['agent_state']).unsqueeze(0).to(self.device)

            q_values = self.model(lob_window, dynamic_state, agent_state)
            action = q_values.argmax(dim=1).item()

        self.model.train()
        return action

    def train_step(self, batch, batch_size: int) -> dict:
        lob_window = torch.FloatTensor(np.stack(batch.state[:, 0])).to(self.device)
        dynamic_state = torch.FloatTensor(np.stack(batch.state[:, 1])).to(self.device)
        agent_state = torch.FloatTensor(np.stack(batch.state[:, 2])).to(self.device)
        actions = torch.LongTensor(batch.action).to(self.device)
        rewards = torch.FloatTensor(batch.reward).to(self.device)
        next_lob_window = torch.FloatTensor(np.stack(batch.next_state[:, 0])).to(self.device)
        next_dynamic_state = torch.FloatTensor(np.stack(batch.next_state[:, 1])).to(self.device)
        next_agent_state = torch.FloatTensor(np.stack(batch.next_state[:, 2])).to(self.device)
        dones = torch.FloatTensor(batch.done).to(self.device)

        current_q_values = self.model(lob_window, dynamic_state, agent_state)
        q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        next_q_values = self.target_model(next_lob_window, next_dynamic_state, next_agent_state)
        next_q_values = next_q_values.max(dim=1)[0]

        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = F.mse_loss(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}
