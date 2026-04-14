"""
Improved Train Script for AttnLOB-DQN
=====================================
支持:
- Dueling DQN 架构
- Double DQN 训练
- LayerNorm 归一化
- 梯度裁剪
- Huber Loss

使用方法:
    # 默认：使用改进版本 (Dueling + Double DQN)
    python train_improved.py

    # 使用原始版本对比
    python train_improved.py --original

    # 指定训练参数
    python train_improved.py --epochs 10 --events 10000
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import time
from dataclasses import replace, dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from collections import deque, namedtuple

# 导入配置和模型
from mm_dqn.config import EnvConfig, ModelConfig, TrainConfig
from mm_dqn.model import DuelingDQNNet, OriginalDQNNet, AttnLOBEncoder
from mm_dqn.env import MarketMakingEnv
from mm_dqn.features import mid_price

# 尝试导入额外模块（如果存在）
try:
    from mm_dqn.china_synthetic import build_ping_an_sz_synthetic_events
    HAS_CHINA_SYNTHETIC = True
except ImportError:
    HAS_CHINA_SYNTHETIC = False

try:
    from mm_dqn.io_events import load_events_file
    HAS_IO_EVENTS = True
except ImportError:
    HAS_IO_EVENTS = False


# ============================================
# Replay Buffer
# ============================================

@dataclass
class ReplayBuffer:
    """经验回放缓冲区"""
    capacity: int = 50_000
    buffer: deque = field(default_factory=lambda: deque(maxlen=50000))
    
    def push(self, lob, dyn, agent, action, reward, next_lob, next_dyn, next_agent, done):
        self.buffer.append({
            'lob': lob, 'dyn': dyn, 'agent': agent,
            'action': action, 'reward': reward,
            'next_lob': next_lob, 'next_dyn': next_dyn, 'next_agent': next_agent,
            'done': done
        })
    
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        return batch
    
    def __len__(self):
        return len(self.buffer)


# ============================================
# Double DQN Agent
# ============================================

class DoubleDQNAgent:
    """
    Double DQN Agent with Dueling Architecture
    - 使用 Online 网络选择动作
    - 使用 Target 网络评估值
    - 减少 Q 值过估计问题
    """
    def __init__(
        self,
        env_cfg: EnvConfig,
        model_cfg: ModelConfig,
        train_cfg: TrainConfig,
        device: str = 'cpu'
    ):
        self.env_cfg = env_cfg
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.device = device
        
        # 创建 Online 网络
        self.model = DuelingDQNNet(
            n_levels=env_cfg.n_levels,
            dynamic_dim=model_cfg.dynamic_dim,
            agent_dim=model_cfg.agent_dim,
            hidden_dim=model_cfg.hidden_dim,
            action_dim=model_cfg.action_dim,
            n_heads=model_cfg.n_heads
        ).to(device)
        
        # 创建 Target 网络
        self.target_model = DuelingDQNNet(
            n_levels=env_cfg.n_levels,
            dynamic_dim=model_cfg.dynamic_dim,
            agent_dim=model_cfg.agent_dim,
            hidden_dim=model_cfg.hidden_dim,
            action_dim=model_cfg.action_dim,
            n_heads=model_cfg.n_heads
        ).to(device)
        
        # 同步参数
        self.target_model.load_state_dict(self.model.state_dict())
        
        # 优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=train_cfg.lr)
        
        # 经验回放缓冲区
        self.buffer = ReplayBuffer(capacity=train_cfg.buffer_size)
        
        # 训练状态
        self.steps = 0
        self.epsilon = lambda: max(
            train_cfg.epsilon_end,
            train_cfg.epsilon_start - self.steps / train_cfg.epsilon_decay_steps
        )
        
        # 用于 EMA loss 计算
        self.last_loss = 0.0
        
        print(f"  ✓ Double DQN Agent created")
        print(f"  ✓ Online parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  ✓ Buffer capacity: {train_cfg.buffer_size:,}")
    
    def act(self, lob, dyn, agent_state, epsilon: float = None) -> int:
        """epsilon-greedy 动作选择"""
        if epsilon is None:
            epsilon = self.epsilon()
        
        if random.random() < epsilon:
            return random.randint(0, self.model_cfg.action_dim - 1)
        
        self.model.eval()
        with torch.no_grad():
            lob_t = torch.FloatTensor(lob).unsqueeze(0).to(self.device)
            dyn_t = torch.FloatTensor(dyn).unsqueeze(0).to(self.device)
            agent_t = torch.FloatTensor(agent_state).unsqueeze(0).to(self.device)
            
            q_values = self.model(lob_t, dyn_t, agent_t)
            action = q_values.argmax(dim=1).item()
        
        self.model.train()
        return action
    
    def act_greedy(self, lob, dyn, agent_state) -> int:
        """贪婪动作选择（用于评估）"""
        return self.act(lob, dyn, agent_state, epsilon=0.0)
    
    def push(self, lob, dyn, agent, action, reward, next_lob, next_dyn, next_agent, done):
        """添加经验到缓冲区"""
        self.buffer.push(lob, dyn, agent, action, reward, next_lob, next_dyn, next_agent, done)
    
    def train_step(self) -> Tuple[float, float, bool]:
        """
        执行一次训练步骤
        返回: (loss, td_error, optimized)
        """
        batch_size = self.train_cfg.batch_size
        warmup = self.steps < self.train_cfg.warmup_steps
        
        if len(self.buffer) < batch_size or warmup:
            return 0.0, 0.0, False
        
        # 采样 batch
        batch = self.buffer.sample(batch_size)
        
        # 转换为张量
        lob = torch.FloatTensor(np.stack([b['lob'] for b in batch])).to(self.device)
        dyn = torch.FloatTensor(np.stack([b['dyn'] for b in batch])).to(self.device)
        agent = torch.FloatTensor(np.stack([b['agent'] for b in batch])).to(self.device)
        actions = torch.LongTensor([b['action'] for b in batch]).to(self.device)
        rewards = torch.FloatTensor([b['reward'] for b in batch]).to(self.device)
        next_lob = torch.FloatTensor(np.stack([b['next_lob'] for b in batch])).to(self.device)
        next_dyn = torch.FloatTensor(np.stack([b['next_dyn'] for b in batch])).to(self.device)
        next_agent = torch.FloatTensor(np.stack([b['next_agent'] for b in batch])).to(self.device)
        dones = torch.FloatTensor([float(b['done']) for b in batch]).to(self.device)
        
        # 计算当前 Q 值
        current_q = self.model(lob, dyn, agent)
        q_values = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN: 使用 Online 网络选择动作，Target 网络评估
        with torch.no_grad():
            next_q_online = self.model(next_lob, next_dyn, next_agent)
            next_actions = next_q_online.argmax(dim=1, keepdim=True)
            
            next_q_target = self.target_model(next_lob, next_dyn, next_agent)
            next_q_values = next_q_target.gather(1, next_actions).squeeze(1)
        
        # TD 目标
        target_q_values = rewards + (self.train_cfg.gamma * next_q_values * (1 - dones))
        
        # Huber Loss（更鲁棒）
        loss = F.smooth_l1_loss(q_values, target_q_values)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        
        self.optimizer.step()
        
        # 更新 EMA loss
        self.last_loss = 0.08 * loss.item() + 0.92 * self.last_loss
        
        return loss.item(), (target_q_values - q_values).abs().mean().item(), True
    
    def update_target(self):
        """硬更新 Target 网络"""
        self.target_model.load_state_dict(self.model.state_dict())


# ============================================
# Original DQNAgent (对比用)
# ============================================

class OriginalDQNAgent:
    """原始 DQNAgent，用于对比实验"""
    def __init__(self, env_cfg, model_cfg, train_cfg, device='cpu'):
        self.env_cfg = env_cfg
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.device = device
        
        self.model = OriginalDQNNet(
            n_levels=env_cfg.n_levels,
            dynamic_dim=model_cfg.dynamic_dim,
            agent_dim=model_cfg.agent_dim,
            hidden_dim=model_cfg.hidden_dim,
            action_dim=model_cfg.action_dim,
            n_heads=model_cfg.n_heads
        ).to(device)
        
        self.target_model = OriginalDQNNet(
            n_levels=env_cfg.n_levels,
            dynamic_dim=model_cfg.dynamic_dim,
            agent_dim=model_cfg.agent_dim,
            hidden_dim=model_cfg.hidden_dim,
            action_dim=model_cfg.action_dim,
            n_heads=model_cfg.n_heads
        ).to(device)
        
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=train_cfg.lr)
        self.buffer = ReplayBuffer(capacity=train_cfg.buffer_size)
        self.steps = 0
        self.epsilon = lambda: max(
            train_cfg.epsilon_end,
            train_cfg.epsilon_start - self.steps / train_cfg.epsilon_decay_steps
        )
        self.last_loss = 0.0
    
    def act(self, lob, dyn, agent_state, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon()
        if random.random() < epsilon:
            return random.randint(0, self.model_cfg.action_dim - 1)
        self.model.eval()
        with torch.no_grad():
            lob_t = torch.FloatTensor(lob).unsqueeze(0).to(self.device)
            dyn_t = torch.FloatTensor(dyn).unsqueeze(0).to(self.device)
            agent_t = torch.FloatTensor(agent_state).unsqueeze(0).to(self.device)
            q_values = self.model(lob_t, dyn_t, agent_t)
            action = q_values.argmax(dim=1).item()
        self.model.train()
        return action
    
    def act_greedy(self, lob, dyn, agent_state):
        return self.act(lob, dyn, agent_state, epsilon=0.0)
    
    def push(self, lob, dyn, agent, action, reward, next_lob, next_dyn, next_agent, done):
        self.buffer.push(lob, dyn, agent, action, reward, next_lob, next_dyn, next_agent, done)
    
    def train_step(self):
        batch_size = self.train_cfg.batch_size
        warmup = self.steps < self.train_cfg.warmup_steps
        if len(self.buffer) < batch_size or warmup:
            return 0.0, 0.0, False
        
        batch = self.buffer.sample(batch_size)
        lob = torch.FloatTensor(np.stack([b['lob'] for b in batch])).to(self.device)
        dyn = torch.FloatTensor(np.stack([b['dyn'] for b in batch])).to(self.device)
        agent = torch.FloatTensor(np.stack([b['agent'] for b in batch])).to(self.device)
        actions = torch.LongTensor([b['action'] for b in batch]).to(self.device)
        rewards = torch.FloatTensor([b['reward'] for b in batch]).to(self.device)
        next_lob = torch.FloatTensor(np.stack([b['next_lob'] for b in batch])).to(self.device)
        next_dyn = torch.FloatTensor(np.stack([b['next_dyn'] for b in batch])).to(self.device)
        next_agent = torch.FloatTensor(np.stack([b['next_agent'] for b in batch])).to(self.device)
        dones = torch.FloatTensor([float(b['done']) for b in batch]).to(self.device)
        
        current_q = self.model(lob, dyn, agent)
        q_values = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 原始 DQN
        with torch.no_grad():
            next_q = self.target_model(next_lob, next_dyn, next_agent).max(dim=1)[0]
        
        target_q_values = rewards + (self.train_cfg.gamma * next_q * (1 - dones))
        loss = F.mse_loss(q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        self.last_loss = 0.08 * loss.item() + 0.92 * self.last_loss
        return loss.item(), (target_q_values - q_values).abs().mean().item(), True
    
    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())


# ============================================
# 数据生成
# ============================================

def make_synthetic_events(cfg: EnvConfig, total_events: int = 30_000, seed: int = 42) -> Dict[str, np.ndarray]:
    """生成合成事件数据"""
    rng = np.random.RandomState(seed)
    
    base = 100.0
    noise = rng.normal(0, 0.01, size=total_events).cumsum()
    mids = base + noise
    spread = np.clip(rng.normal(0.02, 0.005, size=total_events), 0.005, 0.05)
    best_ask = mids + spread / 2
    best_bid = mids - spread / 2
    
    lob = np.zeros((total_events, 4 * cfg.n_levels), dtype=np.float32)
    for t in range(total_events):
        for i in range(cfg.n_levels):
            ask_p = best_ask[t] + i * 0.01
            bid_p = best_bid[t] - i * 0.01
            ask_v = rng.randint(50, 1000)
            bid_v = rng.randint(50, 1000)
            j = 4 * i
            lob[t, j] = ask_p
            lob[t, j + 1] = ask_v
            lob[t, j + 2] = bid_p
            lob[t, j + 3] = bid_v
    
    def rand_vol():
        return rng.randint(0, 100, size=total_events).astype(np.float32)
    
    return {
        "lob": lob,
        "mid": np.asarray([mid_price(a, b) for a, b in zip(best_ask, best_bid)], dtype=np.float32),
        "best_ask": best_ask.astype(np.float32),
        "best_bid": best_bid.astype(np.float32),
        "buy_market_vol": rand_vol(),
        "sell_market_vol": rand_vol(),
        "buy_limit_vol": rand_vol(),
        "sell_limit_vol": rand_vol(),
        "buy_cancel_vol": rand_vol(),
        "sell_cancel_vol": rand_vol(),
    }


def _split_events_timewise(events: Dict[str, np.ndarray], train_ratio: float) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """按时间划分训练/测试集"""
    n = len(events["mid"])
    split_idx = int(n * train_ratio)
    train = {k: v[:split_idx] for k, v in events.items()}
    test = {k: v[split_idx:] for k, v in events.items()}
    return train, test


# ============================================
# 训练和评估函数
# ============================================

def run_episode_train(
    env: MarketMakingEnv,
    agent,
    log: logging.Logger,
    epoch_idx: int,
    total_epochs: int,
    log_interval: float,
    last_log_ts: float,
    reward_since_log: float,
    steps_since_log: int,
) -> Tuple[float, float, float, float, float, int]:
    """运行一个训练 episode"""
    state = env.reset()
    ep_reward = 0.0
    done = False
    
    while not done:
        action = agent.act(state.lob_window, state.dynamic_state, state.agent_state)
        next_state, reward, done, _ = env.step(action)
        
        agent.push(
            state.lob_window,
            state.dynamic_state,
            state.agent_state,
            action,
            reward,
            next_state.lob_window,
            next_state.dynamic_state,
            next_state.agent_state,
            done
        )
        
        loss, _, optimized = agent.train_step()
        agent.steps += 1
        
        if optimized:
            ep_reward += reward
            reward_since_log += reward
            steps_since_log += 1
        
        state = next_state
        
        # 日志输出
        now = time.monotonic()
        if now - last_log_ts >= log_interval:
            dt = now - last_log_ts
            sps = steps_since_log / dt if dt > 0 else 0.0
            log.info(
                "epoch %d/%d | step %d | eps %.3f | buffer %d | reward %.2f | loss %.5f | %.0f step/s",
                epoch_idx + 1, total_epochs, agent.steps, agent.epsilon(),
                len(agent.buffer), reward_since_log, agent.last_loss, sps
            )
            last_log_ts = now
            reward_since_log = 0.0
            steps_since_log = 0
    
    return ep_reward, env.cash, agent.last_loss, last_log_ts, reward_since_log, steps_since_log


def run_episode_eval(env: MarketMakingEnv, agent) -> Tuple[float, float]:
    """运行一个评估 episode"""
    state = env.reset()
    ep_reward = 0.0
    done = False
    
    while not done:
        action = agent.act_greedy(state.lob_window, state.dynamic_state, state.agent_state)
        next_state, reward, done, _ = env.step(action)
        ep_reward += reward
        state = next_state
    
    return ep_reward, env.cash


def run_episode_trace(env: MarketMakingEnv, agent) -> Dict[str, np.ndarray]:
    """运行一个 trace episode，用于绘图"""
    state = env.reset()
    done = False
    mid, best_bid, best_ask = [], [], []
    buy_t, buy_px, sell_t, sell_px = [], [], [], []
    
    while not done:
        action = agent.act_greedy(state.lob_window, state.dynamic_state, state.agent_state)
        next_state, _, done, info = env.step(action)
        
        mid.append(float(info["mid"]))
        best_bid.append(float(info["best_bid"]))
        best_ask.append(float(info["best_ask"]))
        
        if bool(info.get("buy_filled", False)):
            buy_t.append(len(mid) - 1)
            buy_px.append(float(info.get("buy_px", 0)))
        if bool(info.get("sell_filled", False)):
            sell_t.append(len(mid) - 1)
            sell_px.append(float(info.get("sell_px", 0)))
        
        state = next_state
    
    return {
        "mid": np.asarray(mid, dtype=np.float32),
        "best_bid": np.asarray(best_bid, dtype=np.float32),
        "best_ask": np.asarray(best_ask, dtype=np.float32),
        "buy_t": np.asarray(buy_t, dtype=np.int32),
        "buy_px": np.asarray(buy_px, dtype=np.float32),
        "sell_t": np.asarray(sell_t, dtype=np.int32),
        "sell_px": np.asarray(sell_px, dtype=np.float32),
    }


def save_plots(metrics_df: pd.DataFrame, out_dir: Path) -> None:
    """保存训练曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    axes[0, 0].plot(metrics_df["epoch"], metrics_df["train_reward"], color="tab:blue")
    axes[0, 0].set_title("Train Reward")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(metrics_df["epoch"], metrics_df["train_pnl"], color="tab:green")
    axes[0, 1].set_title("Train PnL")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(metrics_df["epoch"], metrics_df["test_pnl"], color="tab:orange")
    axes[1, 0].set_title("Test PnL")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(metrics_df["epoch"], metrics_df["loss_ema"], color="tab:red")
    axes[1, 1].set_title("Loss EMA")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / "training_curves.png", dpi=140)
    plt.close()


def save_quote_plot(trace: Dict[str, np.ndarray], out_dir: Path) -> None:
    """保存报价图"""
    x = np.arange(len(trace["mid"]))
    
    plt.figure(figsize=(14, 5))
    plt.plot(x, trace["mid"], color="gray", linewidth=1.2, label="mid price")
    plt.plot(x, trace["best_bid"], color="red", linewidth=1.0, label="best bid")
    plt.plot(x, trace["best_ask"], color="green", linewidth=1.0, label="best ask")
    
    if len(trace["buy_t"]) > 0:
        plt.scatter(trace["buy_t"], trace["buy_px"], marker="^", s=40, color="tab:blue", label="buy fill")
    if len(trace["sell_t"]) > 0:
        plt.scatter(trace["sell_t"], trace["sell_px"], marker="v", s=40, color="tab:orange", label="sell fill")
    
    plt.title("Market Maker Quoting and Transactions")
    plt.xlabel("Step")
    plt.ylabel("Price")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "market_maker_quotes.png", dpi=150)
    plt.close()


# ============================================
# Main
# ============================================

def main():
    # 命令行参数
    ap = argparse.ArgumentParser(description="Improved AttnLOB-DQN Training")
    ap.add_argument("--data", type=str, default="", help="path to data file")
    ap.add_argument("--events", type=int, default=5_000, help="event count for synthetic data")
    ap.add_argument("--epochs", type=int, default=None, help="override epochs")
    ap.add_argument("--episode-events", type=int, default=None, help="override episode events")
    ap.add_argument("--log-interval", type=float, default=30.0, help="log interval (seconds)")
    ap.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    ap.add_argument("--train-ratio", type=float, default=0.8, help="train/test split ratio")
    ap.add_argument("--eval-every", type=int, default=1, help="evaluate every N epochs")
    ap.add_argument("--out-dir", type=str, default="outputs_improved", help="output directory")
    ap.add_argument("--original", action="store_true", help="use original DQN instead of improved")
    ap.add_argument("--device", type=str, default="auto", help="device: auto/cpu/cuda")
    ap.add_argument("--seed", type=int, default=42, help="random seed")
    
    args = ap.parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(message)s",
        datefmt="%H:%M:%S"
    )
    log = logging.getLogger("train_improved")
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 配置
    env_cfg = EnvConfig()
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()
    
    if args.epochs is not None:
        train_cfg.epochs = args.epochs
    if args.episode_events is not None:
        env_cfg.episode_events = args.episode_events
    
    # 设备
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    # 输出目录
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成或加载数据
    if HAS_IO_EVENTS and args.data:
        try:
            events_all = load_events_file(args.data, n_levels=env_cfg.n_levels)
            log.info("Loaded data from %s (%d events)", args.data, len(events_all["mid"]))
        except Exception as e:
            log.warning("Failed to load data: %s, using synthetic", e)
            events_all = make_synthetic_events(env_cfg, args.events, args.seed)
            log.info("Using synthetic data (%d events)", args.events)
    elif HAS_CHINA_SYNTHETIC:
        events_all = build_ping_an_sz_synthetic_events(
            total_events=args.events, n_levels=env_cfg.n_levels, seed=args.seed
        )
        log.info("Using China-style synthetic data (%d events)", args.events)
    else:
        events_all = make_synthetic_events(env_cfg, args.events, args.seed)
        log.info("Using synthetic data (%d events)", args.events)
    
    # 划分数据集
    train_events, test_events = _split_events_timewise(events_all, args.train_ratio)
    
    # 创建环境
    train_episode_events = min(env_cfg.episode_events, len(train_events["mid"]) - 1)
    test_episode_events = min(env_cfg.episode_events, len(test_events["mid"]) - 1)
    
    env_train = MarketMakingEnv(train_events, replace(env_cfg, episode_events=train_episode_events), model_cfg)
    env_test = MarketMakingEnv(test_events, replace(env_cfg, episode_events=test_episode_events), model_cfg)
    
    # 创建 Agent
    log.info("=" * 60)
    if args.original:
        log.info("Using ORIGINAL DQN (for comparison)")
        agent = OriginalDQNAgent(env_cfg, model_cfg, train_cfg, device)
    else:
        log.info("Using IMPROVED DQN:")
        log.info("  - Dueling Architecture (V(s) + A(s,a))")
        log.info("  - Double DQN Training")
        log.info("  - LayerNorm")
        log.info("  - Gradient Clipping")
        log.info("  - Huber Loss")
        agent = DoubleDQNAgent(env_cfg, model_cfg, train_cfg, device)
    log.info("=" * 60)
    
    log.info(
        "device=%s epochs=%d train_events=%d test_events=%d episode=%d",
        device, train_cfg.epochs, len(train_events["mid"]),
        len(test_events["mid"]), train_episode_events
    )
    
    # 训练循环
    metrics = []
    last_log = time.monotonic()
    reward_since_log = 0.0
    steps_since_log = 0
    
    for epoch in range(train_cfg.epochs):
        t_epoch = time.monotonic()
        
        train_reward, train_pnl, loss_ema, last_log, reward_since_log, steps_since_log = run_episode_train(
            env=env_train,
            agent=agent,
            log=log,
            epoch_idx=epoch,
            total_epochs=train_cfg.epochs,
            log_interval=args.log_interval,
            last_log_ts=last_log,
            reward_since_log=reward_since_log,
            steps_since_log=steps_since_log,
        )
        
        test_reward = np.nan
        test_pnl = np.nan
        if (epoch + 1) % args.eval_every == 0:
            test_reward, test_pnl = run_episode_eval(env_test, agent)
        
        metrics.append({
            "epoch": epoch + 1,
            "train_reward": train_reward,
            "test_reward": test_reward,
            "train_pnl": train_pnl,
            "test_pnl": test_pnl,
            "epsilon": agent.epsilon(),
            "loss_ema": loss_ema,
            "epoch_seconds": time.monotonic() - t_epoch,
        })
        
        log.info(
            "epoch %d/%d (%.1fs) | train_r=%.2f pnl=%.2f | test_r=%s pnl=%s | eps=%.3f | loss=%.5f",
            epoch + 1, train_cfg.epochs, metrics[-1]["epoch_seconds"],
            train_reward, train_pnl,
            "n/a" if np.isnan(test_reward) else f"{test_reward:.2f}",
            "n/a" if np.isnan(test_pnl) else f"{test_pnl:.2f}",
            agent.epsilon(), loss_ema
        )
        
        # 保存检查点
        if (epoch + 1) % 5 == 0:
            metrics_df = pd.DataFrame(metrics)
            metrics_path = out_dir / "metrics.csv"
            metrics_df.to_csv(metrics_path, index=False)
            save_plots(metrics_df, out_dir)
            torch.save(agent.model.state_dict(), out_dir / f"model_epoch_{epoch+1}.pt")
            log.info("Saved checkpoint at epoch %d", epoch + 1)
    
    # 最终保存
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(out_dir / "metrics.csv", index=False)
    save_plots(metrics_df, out_dir)
    trace = run_episode_trace(env_test, agent)
    save_quote_plot(trace, out_dir)
    torch.save(agent.model.state_dict(), out_dir / "model_final.pt")
    
    log.info("=" * 60)
    log.info("Training completed!")
    log.info("Saved to: %s", out_dir)
    log.info("  - metrics.csv")
    log.info("  - training_curves.png")
    log.info("  - market_maker_quotes.png")
    log.info("  - model_final.pt")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
