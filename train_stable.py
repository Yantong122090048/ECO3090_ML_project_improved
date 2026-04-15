"""
=============================================
调优后的训练脚本 (Stable Version)
=============================================
改进点:
1. 更低的初始探索率 (0.5 → 0.01)
2. 更快的探索率衰减
3. 更保守的学习率
4. 每 epoch 评估 3 次取平均
5. 添加 Early Stopping

使用方法:
python train_stable.py --epochs 20
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from collections import deque

from mm_dqn.config import EnvConfig, ModelConfig, TrainConfig
from mm_dqn.model import DuelingDQNNet
from mm_dqn.env import MarketMakingEnv


# ============================================
# Stable Double DQN Agent
# ============================================

class StableDoubleDQNAgent:
    """
    稳定版 Double DQN Agent
    改进点:
    - 更低的初始探索率
    - 更稳定的学习率调度
    - 更好的奖励归一化
    """
    def __init__(self, env_cfg, model_cfg, train_cfg, device='cpu'):
        self.env_cfg = env_cfg
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.device = device
        
        # 创建网络
        self.model = DuelingDQNNet(
            n_levels=env_cfg.n_levels,
            dynamic_dim=model_cfg.dynamic_dim,
            agent_dim=model_cfg.agent_dim,
            hidden_dim=model_cfg.hidden_dim,
            action_dim=model_cfg.action_dim,
            n_heads=model_cfg.n_heads
        ).to(device)
        
        self.target_model = DuelingDQNNet(
            n_levels=env_cfg.n_levels,
            dynamic_dim=model_cfg.dynamic_dim,
            agent_dim=model_cfg.agent_dim,
            hidden_dim=model_cfg.hidden_dim,
            action_dim=model_cfg.action_dim,
            n_heads=model_cfg.n_heads
        ).to(device)
        
        self.target_model.load_state_dict(self.model.state_dict())
        
        # 使用 AdamW 和 weight decay 提高稳定性
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=train_cfg.lr,
            weight_decay=1e-4  # L2 正则化
        )
        
        # 学习率调度
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=10, 
            gamma=0.9  # 每 10 epochs 学习率降低 10%
        )
        
        # Replay Buffer
        self.buffer = deque(maxlen=train_cfg.buffer_size)
        
        self.steps = 0
        self.epsilon = lambda: max(
            train_cfg.epsilon_end,
            train_cfg.epsilon_start * (1 - self.steps / train_cfg.epsilon_decay_steps) ** 2  # 二次衰减
        )
        
        self.last_loss = 0.0
        self.best_test_pnl = float('-inf')
        self.patience_counter = 0
        
        print(f"  ✓ Stable Agent created")
        print(f"  ✓ LR: {train_cfg.lr}, Weight Decay: 1e-4")
    
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
        self.buffer.append({
            'lob': lob, 'dyn': dyn, 'agent': agent,
            'action': action, 'reward': reward,
            'next_lob': next_lob, 'next_dyn': next_dyn, 
            'next_agent': next_agent, 'done': done
        })
    
    def train_step(self):
        batch_size = self.train_cfg.batch_size
        warmup = self.steps < self.train_cfg.warmup_steps
        
        if len(self.buffer) < batch_size or warmup:
            return 0.0, 0.0, False
        
        batch = random.sample(self.buffer, batch_size)
        
        lob = torch.FloatTensor(np.stack([b['lob'] for b in batch])).to(self.device)
        dyn = torch.FloatTensor(np.stack([b['dyn'] for b in batch])).to(self.device)
        agent = torch.FloatTensor(np.stack([b['agent'] for b in batch])).to(self.device)
        actions = torch.LongTensor([b['action'] for b in batch]).to(self.device)
        rewards = torch.FloatTensor([b['reward'] for b in batch]).to(self.device)
        next_lob = torch.FloatTensor(np.stack([b['next_lob'] for b in batch])).to(self.device)
        next_dyn = torch.FloatTensor(np.stack([b['next_dyn'] for b in batch])).to(self.device)
        next_agent = torch.FloatTensor(np.stack([b['next_agent'] for b in batch])).to(self.device)
        dones = torch.FloatTensor([float(b['done']) for b in batch]).to(self.device)
        
        # Reward normalization
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # Double DQN
        current_q = self.model(lob, dyn, agent)
        q_values = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q_online = self.model(next_lob, next_dyn, next_agent)
            next_actions = next_q_online.argmax(dim=1, keepdim=True)
            
            next_q_target = self.target_model(next_lob, next_dyn, next_agent)
            next_q_values = next_q_target.gather(1, next_actions).squeeze(1)
        
        target_q = rewards + (self.train_cfg.gamma * next_q_values * (1 - dones))
        
        # Huber loss
        loss = F.smooth_l1_loss(q_values, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # 更严格的梯度裁剪
        self.optimizer.step()
        
        self.last_loss = 0.08 * loss.item() + 0.92 * self.last_loss
        
        return loss.item(), (target_q - q_values).abs().mean().item(), True
    
    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())


# ============================================
# 数据生成
# ============================================

def make_synthetic_events(cfg, total_events=30000, seed=42):
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
    
    rand_vol = lambda: rng.randint(0, 100, size=total_events).astype(np.float32)
    
    return {
        "lob": lob,
        "mid": np.asarray([(a + b) / 2 for a, b in zip(best_ask, best_bid)], dtype=np.float32),
        "best_ask": best_ask.astype(np.float32),
        "best_bid": best_bid.astype(np.float32),
        "buy_market_vol": rand_vol(),
        "sell_market_vol": rand_vol(),
        "buy_limit_vol": rand_vol(),
        "sell_limit_vol": rand_vol(),
        "buy_cancel_vol": rand_vol(),
        "sell_cancel_vol": rand_vol(),
    }


def _split_events(events, ratio=0.8):
    n = len(events["mid"])
    split_idx = int(n * ratio)
    train = {k: v[:split_idx] for k, v in events.items()}
    test = {k: v[split_idx:] for k, v in events.items()}
    return train, test


# ============================================
# 训练函数
# ============================================

def run_episode(env, agent, is_train=True, epsilon=None):
    state = env.reset()
    ep_reward = 0.0
    ep_pnl = 0.0
    done = False
    
    while not done:
        if is_train:
            action = agent.act(state.lob_window, state.dynamic_state, state.agent_state, epsilon)
        else:
            action = agent.act_greedy(state.lob_window, state.dynamic_state, state.agent_state)
        
        next_state, reward, done, _ = env.step(action)
        
        if is_train:
            agent.push(
                state.lob_window, state.dynamic_state, state.agent_state,
                action, reward,
                next_state.lob_window, next_state.dynamic_state, next_state.agent_state,
                done
            )
            loss, _, optimized = agent.train_step()
            agent.steps += 1
        
        ep_reward += reward
        ep_pnl += reward
        state = next_state
    
    return ep_reward, env.cash


def run_multiple_eval(env, agent, n_eval=3):
    """多次评估取平均，更稳定"""
    rewards = []
    pnls = []
    for _ in range(n_eval):
        r, pnl = run_episode(env, agent, is_train=False)
        rewards.append(r)
        pnls.append(pnl)
    return np.mean(rewards), np.mean(pnls)


def main():
    parser = argparse.ArgumentParser(description='Stable DQN Training')
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--events", type=int, default=10000)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--out-dir", type=str, default="outputs_stable")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    logging.basicConfig(level="INFO", format="%(asctime)s | %(message)s")
    log = logging.getLogger("stable_train")
    
    # 配置 - 使用更稳定的超参数
    env_cfg = EnvConfig()
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()
    
    # 🔧 关键修改：更稳定的超参数
    train_cfg.epsilon_start = 0.5   # 从 1.0 降到 0.5
    train_cfg.epsilon_end = 0.01   # 从 0.05 降到 0.01
    train_cfg.epsilon_decay_steps = 5000  # 从 10000 降到 5000
    train_cfg.lr = 5e-5           # 从 1e-4 降到 5e-5
    train_cfg.batch_size = 32     # 从 64 降到 32
    train_cfg.warmup_steps = 500  # 从 200 增加到 500
    
    train_cfg.epochs = args.epochs
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    log.info("=" * 60)
    log.info("🚀 Stable Training Configuration:")
    log.info(f"  epsilon_start: {train_cfg.epsilon_start}")
    log.info(f"  epsilon_end: {train_cfg.epsilon_end}")
    log.info(f"  lr: {train_cfg.lr}")
    log.info(f"  batch_size: {train_cfg.batch_size}")
    log.info(f"  warmup_steps: {train_cfg.warmup_steps}")
    log.info("=" * 60)
    
    # 生成数据
    events_all = make_synthetic_events(env_cfg, args.events, args.seed)
    train_events, test_events = _split_events(events_all, 0.8)
    
    train_ep = min(env_cfg.episode_events, len(train_events["mid"]) - 1)
    test_ep = min(env_cfg.episode_events, len(test_events["mid"]) - 1)
    
    env_train = MarketMakingEnv(train_events, replace(env_cfg, episode_events=train_ep), model_cfg)
    env_test = MarketMakingEnv(test_events, replace(env_cfg, episode_events=test_ep), model_cfg)
    
    agent = StableDoubleDQNAgent(env_cfg, model_cfg, train_cfg, device)
    
    metrics = []
    best_pnl = float('-inf')
    patience_counter = 0
    
    for epoch in range(train_cfg.epochs):
        t_start = time.time()
        
        # 训练
        train_reward, train_pnl = run_episode(
            env_train, agent, is_train=True, epsilon=agent.epsilon()
        )
        
        # 评估 (多次取平均)
        if (epoch + 1) % args.eval_every == 0:
            test_reward, test_pnl = run_multiple_eval(env_test, agent, n_eval=3)
        else:
            test_reward, test_pnl = float('nan'), float('nan')
        
        # 学习率调度
        agent.scheduler.step()
        
        # Early stopping 检查
        if not np.isnan(test_pnl):
            if test_pnl > best_pnl:
                best_pnl = test_pnl
                patience_counter = 0
                torch.save(agent.model.state_dict(), out_dir / "best_model.pt")
            else:
                patience_counter += 1
        
        metrics.append({
            "epoch": epoch + 1,
            "train_reward": train_reward,
            "test_reward": test_reward,
            "train_pnl": train_pnl,
            "test_pnl": test_pnl,
            "epsilon": agent.epsilon(),
            "loss_ema": agent.last_loss,
            "lr": agent.optimizer.param_groups[0]['lr'],
            "epoch_seconds": time.time() - t_start,
        })
        
        log.info(
            "epoch %d/%d | eps %.3f | lr %.6f | train_pnl %.2f | test_pnl %.2f | loss %.4f",
            epoch + 1, train_cfg.epochs, agent.epsilon(), 
            agent.optimizer.param_groups[0]['lr'],
            train_pnl, test_pnl if not np.isnan(test_pnl) else 0,
            agent.last_loss
        )
        
        # Early stopping
        if patience_counter >= args.patience:
            log.info(f"Early stopping at epoch {epoch + 1} (no improvement for {args.patience} epochs)")
            break
    
    # 保存结果
    df = pd.DataFrame(metrics)
    df.to_csv(out_dir / "metrics.csv", index=False)
    
    log.info("=" * 60)
    log.info("✅ Training completed!")
    log.info(f"Best Test PnL: {best_pnl:.4f}")
    log.info(f"Saved to: {out_dir}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
