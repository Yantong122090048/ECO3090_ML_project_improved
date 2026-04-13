from __future__ import annotations

import numpy as np


def mid_price(best_ask: float, best_bid: float) -> float:
    return 0.5 * (best_ask + best_bid)


def stationary_lob_window(lob_window: np.ndarray) -> np.ndarray:
    # convert prices to relative changes and normalize volumes
    x = lob_window.copy()
    prices = x[:, 0::2]
    vols = x[:, 1::2]
    base = prices[:, 0:1] + 1e-8
    prices = (prices - base) / base
    prices = (prices - prices.mean()) / (prices.std() + 1e-8)
    vols = vols / (np.max(vols) + 1e-8)
    out = np.empty_like(x)
    out[:, 0::2] = prices
    out[:, 1::2] = vols
    return out


def realized_volatility(prices: np.ndarray) -> float:
    if len(prices) < 2:
        return 0.0
    log_r = np.diff(np.log(np.clip(prices, 1e-8, None)))
    return float(np.sqrt(np.sum(log_r * log_r)))


def rsi(prices: np.ndarray) -> float:
    if len(prices) < 2:
        return 0.5
    d = np.diff(prices)
    gain = np.clip(d, 0.0, None).sum()
    loss = np.clip(-d, 0.0, None).sum()
    return float(gain / (gain + loss + 1e-8))


def osi(values_buy: np.ndarray, values_sell: np.ndarray) -> float:
    b = float(np.sum(values_buy))
    s = float(np.sum(values_sell))
    return (b - s) / (b + s + 1e-8)


def compute_dynamic_features(
    mids: np.ndarray,
    buy_market_vol: np.ndarray,
    sell_market_vol: np.ndarray,
    buy_limit_vol: np.ndarray,
    sell_limit_vol: np.ndarray,
    buy_cancel_vol: np.ndarray,
    sell_cancel_vol: np.ndarray,
) -> np.ndarray:
    # 18 osi + 3 rv + 3 rsi
    horizons = [10, 60, 300]
    feats = []
    for h in horizons:
        sl = slice(max(0, len(mids) - h), len(mids))
        feats.extend(
            [
                osi(buy_market_vol[sl], sell_market_vol[sl]),
                osi(buy_limit_vol[sl], sell_limit_vol[sl]),
                osi(buy_cancel_vol[sl], sell_cancel_vol[sl]),
                osi((buy_market_vol[sl] > 0).astype(float), (sell_market_vol[sl] > 0).astype(float)),
                osi((buy_limit_vol[sl] > 0).astype(float), (sell_limit_vol[sl] > 0).astype(float)),
                osi((buy_cancel_vol[sl] > 0).astype(float), (sell_cancel_vol[sl] > 0).astype(float)),
            ]
        )
    for h in horizons:
        sl = slice(max(0, len(mids) - h), len(mids))
        feats.append(realized_volatility(mids[sl]))
    for h in horizons:
        sl = slice(max(0, len(mids) - h), len(mids))
        feats.append(rsi(mids[sl]))
    return np.asarray(feats, dtype=np.float32)
