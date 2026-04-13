from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from mm_dqn.config import EnvConfig, ModelConfig
from mm_dqn.features import compute_dynamic_features, mid_price, stationary_lob_window


@dataclass
class State:
    lob_window: np.ndarray
    dynamic_state: np.ndarray
    agent_state: np.ndarray


class MarketMakingEnv:
    def __init__(self, lob_events: Dict[str, np.ndarray], env_cfg: EnvConfig, model_cfg: ModelConfig):
        self.data = lob_events
        self.cfg = env_cfg
        self.mcfg = model_cfg
        self._t = 0
        self.cash = 0.0
        self.inventory = 0.0
        self.done = False
        self.rng = np.random.default_rng(42)

        self.a1_bins = np.linspace(0.0, 1.0, self.mcfg.action_bins_a1)
        self.a2_bins = np.linspace(0.0, 1.0, self.mcfg.action_bins_a2)

    def _decode_action(self, action_id: int) -> Tuple[float, float]:
        i = action_id // self.mcfg.action_bins_a2
        j = action_id % self.mcfg.action_bins_a2
        return float(self.a1_bins[i]), float(self.a2_bins[j])

    def reset(self) -> State:
        self._t = self.cfg.window_size
        self.cash = 0.0
        self.inventory = 0.0
        self.done = False
        return self._build_state()

    def _build_state(self) -> State:
        s = self._t - self.cfg.window_size
        e = self._t
        lob_window = self.data["lob"][s:e]
        mids = self.data["mid"][:e]
        dyn = compute_dynamic_features(
            mids=mids,
            buy_market_vol=self.data["buy_market_vol"][:e],
            sell_market_vol=self.data["sell_market_vol"][:e],
            buy_limit_vol=self.data["buy_limit_vol"][:e],
            sell_limit_vol=self.data["sell_limit_vol"][:e],
            buy_cancel_vol=self.data["buy_cancel_vol"][:e],
            sell_cancel_vol=self.data["sell_cancel_vol"][:e],
        )
        agent = np.asarray(
            [
                self.inventory / (self.cfg.max_inventory_units * self.cfg.min_trade_unit + 1e-8),
                self._t / max(1, self.cfg.episode_events),
            ],
            dtype=np.float32,
        )
        return State(stationary_lob_window(lob_window).astype(np.float32), dyn.astype(np.float32), agent)

    def _execute_quotes(self, bid: float, ask: float) -> Tuple[float, float, Dict]:
        best_ask = float(self.data["best_ask"][self._t])
        best_bid = float(self.data["best_bid"][self._t])
        vol = float(self.cfg.min_trade_unit)
        tp = 0.0
        mid = float(self.data["mid"][self._t])
        spread = max(1e-6, best_ask - best_bid)
        sell_mkt = float(self.data["sell_market_vol"][self._t])
        buy_mkt = float(self.data["buy_market_vol"][self._t])

        # crossing quotes fill immediately
        fill_buy_cross = bid >= best_ask
        fill_sell_cross = ask <= best_bid

        # passive fills near top of book
        buy_touch = np.clip((bid - best_bid) / spread, 0.0, 1.0)
        sell_touch = np.clip((best_ask - ask) / spread, 0.0, 1.0)
        buy_flow = np.clip(sell_mkt / 80.0, 0.0, 1.0)
        sell_flow = np.clip(buy_mkt / 80.0, 0.0, 1.0)
        p_buy = 0.01 + 0.34 * buy_touch + 0.45 * buy_flow
        p_sell = 0.01 + 0.34 * sell_touch + 0.45 * sell_flow
        fill_buy_passive = (bid >= best_bid) and (self.rng.random() < p_buy)
        fill_sell_passive = (ask <= best_ask) and (self.rng.random() < p_sell)

        fill_buy = fill_buy_cross or fill_buy_passive
        fill_sell = fill_sell_cross or fill_sell_passive

        buy_px = np.nan
        sell_px = np.nan
        buy_filled = False
        sell_filled = False
        if fill_buy and self.inventory < self.cfg.max_inventory_units * vol:
            buy_px = best_ask if fill_buy_cross else bid
            self.cash -= buy_px * vol
            self.inventory += vol
            tp += vol * (mid - buy_px)
            buy_filled = True
        if fill_sell and self.inventory > -self.cfg.max_inventory_units * vol:
            sell_px = best_bid if fill_sell_cross else ask
            self.cash += sell_px * vol
            self.inventory -= vol
            tp += (-vol) * (mid - sell_px)
            sell_filled = True
        return tp, self.inventory, {"buy_filled": buy_filled, "sell_filled": sell_filled, "buy_px": buy_px, "sell_px": sell_px}

    def step(self, action_id: int) -> Tuple[State, float, bool, Dict]:
        if self.done:
            raise RuntimeError("episode already done")
        prev_mid = float(self.data["mid"][self._t - 1])
        cur_mid = float(self.data["mid"][self._t])
        prev_value = self.cash + self.inventory * prev_mid

        a1, a2 = self._decode_action(action_id)
        bias = a1 * self.cfg.max_bias
        spread = max(1e-6, a2 * self.cfg.max_spread)
        pr = cur_mid - np.sign(self.inventory) * bias
        bid = pr - spread / 2.0
        ask = pr + spread / 2.0

        tp, _, exec_info = self._execute_quotes(bid=bid, ask=ask)
        cur_value = self.cash + self.inventory * cur_mid
        dp = (cur_value - prev_value) - max(0.0, self.cfg.eta * (cur_value - prev_value))
        ip = self.cfg.zeta * (self.inventory ** 2)
        reward = float(dp + tp - ip)

        self._t += 1
        if self._t >= min(len(self.data["mid"]) - 1, self.cfg.episode_events):
            # close position at end of episode
            self.cash += self.inventory * cur_mid
            self.inventory = 0.0
            self.done = True
        next_state = self._build_state() if not self.done else State(
            lob_window=np.zeros((self.cfg.window_size, 4 * self.cfg.n_levels), dtype=np.float32),
            dynamic_state=np.zeros((self.mcfg.dynamic_dim,), dtype=np.float32),
            agent_state=np.zeros((2,), dtype=np.float32),
        )
        return next_state, reward, self.done, {
            "t": self._t - 1,
            "mid": cur_mid,
            "best_bid": float(self.data["best_bid"][self._t - 1]),
            "best_ask": float(self.data["best_ask"][self._t - 1]),
            "bid": bid,
            "ask": ask,
            "tp": tp,
            "dp": dp,
            "ip": ip,
            **exec_info,
        }
