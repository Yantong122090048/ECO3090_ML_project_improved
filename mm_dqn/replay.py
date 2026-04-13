from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, List

import numpy as np


@dataclass
class Transition:
    lob: np.ndarray
    dyn: np.ndarray
    agent: np.ndarray
    action: int
    reward: float
    next_lob: np.ndarray
    next_dyn: np.ndarray
    next_agent: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf: Deque[Transition] = deque(maxlen=capacity)

    def push(self, t: Transition) -> None:
        self.buf.append(t)

    def __len__(self) -> int:
        return len(self.buf)

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buf, batch_size)
