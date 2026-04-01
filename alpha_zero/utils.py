"""
Utility helpers for the AlphaZero implementation.
"""

import logging
import sys
import time
import numpy as np
from typing import List, Tuple


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a clean timestamped format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


class AverageMeter:
    """Tracks a running mean — useful for logging loss values."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self) -> str:
        return f"{self.avg:.4f}"


class Timer:
    """Simple wall-clock timer."""

    def __init__(self):
        self._start = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self._start

    def reset(self):
        self._start = time.perf_counter()

    def __str__(self) -> str:
        e = self.elapsed()
        if e < 60:
            return f"{e:.1f}s"
        return f"{e/60:.1f}m"


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(x - x.max())
    return e / e.sum()


def policy_from_counts(counts: np.ndarray, temperature: float) -> np.ndarray:
    """Convert visit counts to a policy distribution at the given temperature."""
    if temperature == 0:
        policy = np.zeros_like(counts, dtype=np.float64)
        policy[np.argmax(counts)] = 1.0
        return policy
    counts = counts.astype(np.float64)
    counts **= (1.0 / temperature)
    total = counts.sum()
    if total == 0:
        return np.ones_like(counts) / len(counts)
    return counts / total


def encode_board_planes(
    arrays: List[np.ndarray],
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    Stack a list of 2-D boolean/float arrays into a (C, H, W) tensor.

    Convenience function for building state representations:
        planes = encode_board_planes([my_pieces, opp_pieces, kings, ...])
    """
    return np.stack(arrays, axis=0).astype(dtype)


def print_progress_bar(
    iteration: int,
    total: int,
    prefix: str = "",
    suffix: str = "",
    length: int = 40,
) -> None:
    """Print a simple ASCII progress bar to stdout."""
    filled = int(length * iteration // total)
    bar = "=" * filled + "-" * (length - filled)
    pct = 100.0 * iteration / total
    print(f"\r{prefix} [{bar}] {pct:.1f}% {suffix}", end="", flush=True)
    if iteration == total:
        print()
