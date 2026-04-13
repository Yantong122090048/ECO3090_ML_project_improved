from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, Optional, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from mm_dqn.agent import DQNAgent
from mm_dqn.china_synthetic import build_ping_an_sz_synthetic_events
from mm_dqn.config import EnvConfig, ModelConfig, TrainConfig
from mm_dqn.env import MarketMakingEnv
from mm_dqn.features import mid_price
from mm_dqn.io_events import load_events_file
from mm_dqn.replay import Transition


def _exit_if_numpy_torch_mismatch() -> None:
    try:
        torch.from_numpy(np.zeros((1,), dtype=np.float32))
    except RuntimeError as e:
        if "Numpy is not available" in str(e) or "_ARRAY_API" in str(e):
            print(
                "PyTorch and NumPy are ABI-incompatible. Fix one of:\n"
                "  pip install 'numpy>=1.26,<2'\n"
                "  or upgrade PyTorch to a release that supports NumPy 2.x (e.g. torch>=2.3).\n",
                file=sys.stderr,
            )
            raise SystemExit(1) from e


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_synthetic_events(cfg: EnvConfig, total_events: int = 30_000) -> Dict[str, np.ndarray]:
    base = 100.0
    noise = np.random.normal(0, 0.01, size=total_events).cumsum()
    mids = base + noise
    spread = np.clip(np.random.normal(0.02, 0.005, size=total_events), 0.005, 0.05)
    best_ask = mids + spread / 2
    best_bid = mids - spread / 2
    lob = np.zeros((total_events, 4 * cfg.n_levels), dtype=np.float32)
    for t in range(total_events):
        for i in range(cfg.n_levels):
            ask_p = best_ask[t] + i * 0.01
            bid_p = best_bid[t] - i * 0.01
            ask_v = np.random.randint(50, 1000)
            bid_v = np.random.randint(50, 1000)
            j = 4 * i
            lob[t, j] = ask_p
            lob[t, j + 1] = ask_v
            lob[t, j + 2] = bid_p
            lob[t, j + 3] = bid_v
    rand_vol = lambda: np.random.randint(0, 100, size=total_events).astype(np.float32)
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


def _resolve_data_path(cli_path: str) -> Optional[Path]:
    if cli_path.strip():
        return Path(cli_path)
    for name in ("china_sz000001_pingan_synthetic.csv", "china_sz000001_pingan_synthetic.npz"):
        p = Path("data") / name
        if p.is_file():
            return p
    return None


def _split_events_timewise(events: Dict[str, np.ndarray], train_ratio: float) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    n = len(events["mid"])
    split_idx = int(n * train_ratio)
    train = {k: v[:split_idx] for k, v in events.items()}
    test = {k: v[split_idx:] for k, v in events.items()}
    return train, test


def _run_episode_train(
    env: MarketMakingEnv,
    agent: DQNAgent,
    loss_ema: Optional[float],
    log: logging.Logger,
    epoch_idx: int,
    total_epochs: int,
    log_interval: float,
    last_log_ts: float,
    reward_since_log: float,
    steps_since_log: int,
) -> Tuple[float, float, Optional[float], float, float, int]:
    state = env.reset()
    ep_reward = 0.0
    done = False
    while not done:
        action = agent.act(state.lob_window, state.dynamic_state, state.agent_state)
        next_state, reward, done, _ = env.step(action)
        agent.push(
            Transition(
                lob=state.lob_window,
                dyn=state.dynamic_state,
                agent=state.agent_state,
                action=action,
                reward=reward,
                next_lob=next_state.lob_window,
                next_dyn=next_state.dynamic_state,
                next_agent=next_state.agent_state,
                done=done,
            )
        )
        loss, _, optimized = agent.train_step()
        agent.steps += 1
        if optimized:
            loss_ema = loss if loss_ema is None else 0.08 * loss + 0.92 * loss_ema
        ep_reward += reward
        reward_since_log += reward
        steps_since_log += 1
        state = next_state

        now = time.monotonic()
        if now - last_log_ts >= log_interval:
            dt = now - last_log_ts
            sps = steps_since_log / dt if dt > 0 else 0.0
            le = f"{loss_ema:.5f}" if loss_ema is not None else "n/a"
            log.info(
                "epoch %d/%d | global_step %d | eps %.3f | buffer %d | reward_window %.2f | loss_ema %s | %.0f step/s",
                epoch_idx + 1,
                total_epochs,
                agent.steps,
                agent.epsilon(),
                len(agent.buffer),
                reward_since_log,
                le,
                sps,
            )
            last_log_ts = now
            reward_since_log = 0.0
            steps_since_log = 0

    return ep_reward, env.cash, loss_ema, last_log_ts, reward_since_log, steps_since_log


def _run_episode_eval(env: MarketMakingEnv, agent: DQNAgent) -> Tuple[float, float]:
    state = env.reset()
    ep_reward = 0.0
    done = False
    while not done:
        action = agent.act_greedy(state.lob_window, state.dynamic_state, state.agent_state)
        next_state, reward, done, _ = env.step(action)
        ep_reward += reward
        state = next_state
    return ep_reward, env.cash


def _run_episode_eval_trace(env: MarketMakingEnv, agent: DQNAgent) -> Dict[str, np.ndarray]:
    state = env.reset()
    done = False
    mid, best_bid, best_ask = [], [], []
    buy_fill_t, buy_fill_px = [], []
    sell_fill_t, sell_fill_px = [], []
    while not done:
        action = agent.act_greedy(state.lob_window, state.dynamic_state, state.agent_state)
        next_state, _, done, info = env.step(action)
        mid.append(float(info["mid"]))
        best_bid.append(float(info["best_bid"]))
        best_ask.append(float(info["best_ask"]))
        if bool(info["buy_filled"]):
            buy_fill_t.append(len(mid) - 1)
            buy_fill_px.append(float(info["buy_px"]))
        if bool(info["sell_filled"]):
            sell_fill_t.append(len(mid) - 1)
            sell_fill_px.append(float(info["sell_px"]))
        state = next_state
    return {
        "mid": np.asarray(mid, dtype=np.float32),
        "best_bid": np.asarray(best_bid, dtype=np.float32),
        "best_ask": np.asarray(best_ask, dtype=np.float32),
        "buy_fill_t": np.asarray(buy_fill_t, dtype=np.int32),
        "buy_fill_px": np.asarray(buy_fill_px, dtype=np.float32),
        "sell_fill_t": np.asarray(sell_fill_t, dtype=np.int32),
        "sell_fill_px": np.asarray(sell_fill_px, dtype=np.float32),
    }


def _save_plots(metrics_df: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(12, 8))
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(metrics_df["epoch"], metrics_df["train_reward"], label="train reward", color="tab:blue")
    ax1.set_title("train reward")
    ax1.legend()

    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(metrics_df["epoch"], metrics_df["train_pnl"], label="train pnl", color="tab:green")
    ax2.set_title("train pnl")
    ax2.legend()

    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(metrics_df["epoch"], metrics_df["test_pnl"], label="test pnl", color="tab:orange")
    ax3.set_title("test pnl")
    ax3.legend()

    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(metrics_df["epoch"], metrics_df["loss_ema"], label="loss ema", color="tab:red")
    ax4.set_title("loss ema")
    ax4.legend()

    plt.tight_layout()
    plt.savefig(out_dir / "training_curves.png", dpi=140)
    plt.close()


def _save_quote_plot(trace: Dict[str, np.ndarray], out_dir: Path) -> None:
    x = np.arange(len(trace["mid"]))
    plt.figure(figsize=(14, 5))
    plt.plot(x, trace["mid"], color="gray", linewidth=1.2, label="mid price")
    plt.plot(x, trace["best_bid"], color="red", linewidth=1.0, label="best bid")
    plt.plot(x, trace["best_ask"], color="green", linewidth=1.0, label="best ask")
    if trace["buy_fill_t"].size > 0:
        plt.scatter(trace["buy_fill_t"], trace["buy_fill_px"], marker="^", s=40, color="tab:blue", label="buy fill")
    if trace["sell_fill_t"].size > 0:
        plt.scatter(trace["sell_fill_t"], trace["sell_fill_px"], marker="v", s=40, color="tab:orange", label="sell fill")
    plt.title("market maker quoting and transactions (test out-of-sample)")
    plt.xlabel("step")
    plt.ylabel("price")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_dir / "market_maker_quotes.png", dpi=150)
    plt.close()


def main() -> None:
    _exit_if_numpy_torch_mismatch()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="", help="path to .csv or .npz")
    ap.add_argument("--events", type=int, default=5_000, help="in-memory event count when no data file")
    ap.add_argument("--epochs", type=int, default=None, help="override TrainConfig.epochs")
    ap.add_argument("--episode-events", type=int, default=None, help="override EnvConfig.episode_events")
    ap.add_argument("--log-interval", type=float, default=30.0, help="seconds between progress logs")
    ap.add_argument("--log-level", type=str, default="INFO", choices=("DEBUG", "INFO", "WARNING"))
    ap.add_argument("--train-ratio", type=float, default=0.8, help="time-based split for train/test")
    ap.add_argument("--test-ratio", type=float, default=None, help="optional override for out-of-sample test ratio")
    ap.add_argument("--eval-every", type=int, default=1, help="evaluate on test every N epochs")
    ap.add_argument("--out-dir", type=str, default="outputs", help="where to save metrics and plots")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    log = logging.getLogger("train")

    env_cfg = EnvConfig()
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()
    if args.epochs is not None:
        train_cfg.epochs = args.epochs
    if args.episode_events is not None:
        env_cfg.episode_events = args.episode_events
    set_seed(train_cfg.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = _resolve_data_path(args.data)
    if data_path is not None and data_path.is_file():
        events_all = load_events_file(str(data_path), n_levels=env_cfg.n_levels)
        log.info("loaded data from %s (%d events)", data_path, len(events_all["mid"]))
    else:
        events_all = build_ping_an_sz_synthetic_events(total_events=args.events, n_levels=env_cfg.n_levels, seed=train_cfg.seed)
        log.info("using in-memory china-style synthetic data (%d events)", args.events)

    if args.test_ratio is not None:
        if not (0.0 < args.test_ratio < 1.0):
            raise ValueError("--test-ratio must be in (0, 1)")
        train_ratio = 1.0 - args.test_ratio
    else:
        train_ratio = args.train_ratio
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("--train-ratio must be in (0, 1)")

    train_events, test_events = _split_events_timewise(events_all, train_ratio)
    min_required = env_cfg.window_size + 2
    if len(train_events["mid"]) < min_required or len(test_events["mid"]) < min_required:
        raise ValueError("train/test split is too short. increase event count or adjust --train-ratio")

    train_episode_events = min(env_cfg.episode_events, len(train_events["mid"]) - 1)
    test_episode_events = min(env_cfg.episode_events, len(test_events["mid"]) - 1)
    env_train = MarketMakingEnv(train_events, replace(env_cfg, episode_events=train_episode_events), model_cfg)
    env_test = MarketMakingEnv(test_events, replace(env_cfg, episode_events=test_episode_events), model_cfg)
    agent = DQNAgent(env_cfg, model_cfg, train_cfg, device=device)

    log.info(
        "device=%s epochs=%d train_events=%d test_events=%d episode(train/test)=%d/%d",
        device,
        train_cfg.epochs,
        len(train_events["mid"]),
        len(test_events["mid"]),
        train_episode_events,
        test_episode_events,
    )

    loss_ema: Optional[float] = None
    last_log = time.monotonic()
    reward_since_log = 0.0
    steps_since_log = 0
    metrics = []

    for epoch in range(train_cfg.epochs):
        t_epoch = time.monotonic()
        train_reward, train_pnl, loss_ema, last_log, reward_since_log, steps_since_log = _run_episode_train(
            env=env_train,
            agent=agent,
            loss_ema=loss_ema,
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
            test_reward, test_pnl = _run_episode_eval(env_test, agent)

        metrics.append(
            {
                "epoch": epoch + 1,
                "train_reward": train_reward,
                "test_reward": test_reward,
                "train_pnl": train_pnl,
                "test_pnl": test_pnl,
                "epsilon": agent.epsilon(),
                "loss_ema": np.nan if loss_ema is None else loss_ema,
                "epoch_seconds": time.monotonic() - t_epoch,
            }
        )

        log.info(
            "epoch %d/%d done in %.1fs | train_reward=%.2f train_pnl=%.2f | test_reward=%s test_pnl=%s | epsilon=%.3f | loss_ema=%s",
            epoch + 1,
            train_cfg.epochs,
            metrics[-1]["epoch_seconds"],
            train_reward,
            train_pnl,
            "n/a" if np.isnan(test_reward) else f"{test_reward:.2f}",
            "n/a" if np.isnan(test_pnl) else f"{test_pnl:.2f}",
            agent.epsilon(),
            "n/a" if loss_ema is None else f"{loss_ema:.5f}",
        )

    metrics_df = pd.DataFrame(metrics)
    metrics_path = out_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    _save_plots(metrics_df, out_dir)
    trace = _run_episode_eval_trace(env_test, agent)
    _save_quote_plot(trace, out_dir)

    torch.save(agent.q.state_dict(), out_dir / "dqn_market_maker.pt")
    log.info("saved weights: %s", out_dir / "dqn_market_maker.pt")
    log.info("saved metrics: %s", metrics_path)
    log.info("saved plots: %s", out_dir / "training_curves.png")
    log.info("saved quote plot: %s", out_dir / "market_maker_quotes.png")


if __name__ == "__main__":
    main()
