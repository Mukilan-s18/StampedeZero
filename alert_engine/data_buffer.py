"""
StampedeZero - Time-Series Data Buffer
=======================================
Implements a high-performance rolling window buffer using collections.deque
for storing crowd density readings over a configurable time horizon.

This module serves as the "memory" of the prediction engine — it maintains
a fixed-size FIFO queue of (timestamp, crowd_count) tuples, automatically
discarding the oldest entries when the buffer is full.

Design Decisions:
    - deque(maxlen=N) guarantees O(1) append and O(1) memory ceiling
    - Tuples are used over dicts for lower memory footprint at high frequency
    - Thread-safety note: deque.append() is atomic in CPython (GIL-protected)
"""

from collections import deque
import time
from typing import Optional, Tuple, List


class CrowdDataBuffer:
    """
    Fixed-size rolling window buffer for crowd density time-series data.

    Each entry is a (timestamp, crowd_count) tuple. When the buffer reaches
    its maximum capacity, the oldest entries are silently discarded (FIFO).

    Args:
        buffer_size: Maximum number of data points to retain.
                     At 1 update/sec, buffer_size=60 stores the last 60 seconds.
    """

    def __init__(self, buffer_size: int = 60):
        self._buffer: deque = deque(maxlen=buffer_size)
        self._total_updates: int = 0

    def push(self, crowd_count: int, timestamp: Optional[float] = None) -> None:
        """
        Append a new crowd count reading to the buffer.

        Args:
            crowd_count: Current number of people detected in the zone.
            timestamp:   Unix timestamp of the reading. Defaults to time.time().
        """
        if timestamp is None:
            timestamp = time.time()
        self._buffer.append((timestamp, crowd_count))
        self._total_updates += 1

    def get_data(self) -> List[Tuple[float, int]]:
        """Return a copy of the buffer contents as a list of (timestamp, count) tuples."""
        return list(self._buffer)

    def get_timestamps(self) -> List[float]:
        """Extract only the timestamps from all buffered entries."""
        return [entry[0] for entry in self._buffer]

    def get_counts(self) -> List[int]:
        """Extract only the crowd counts from all buffered entries."""
        return [entry[1] for entry in self._buffer]

    def get_latest(self) -> Optional[Tuple[float, int]]:
        """Return the most recent (timestamp, count) entry, or None if buffer is empty."""
        return self._buffer[-1] if self._buffer else None

    @property
    def size(self) -> int:
        """Current number of entries in the buffer."""
        return len(self._buffer)

    @property
    def capacity(self) -> int:
        """Maximum number of entries the buffer can hold."""
        return self._buffer.maxlen

    @property
    def is_ready(self) -> bool:
        """True if the buffer has enough data points (>=10) for meaningful prediction."""
        return len(self._buffer) >= 10

    @property
    def total_updates(self) -> int:
        """Total number of push() calls since initialization (includes discarded entries)."""
        return self._total_updates

    def clear(self) -> None:
        """Flush all entries from the buffer."""
        self._buffer.clear()
        self._total_updates = 0

    def __len__(self) -> int:
        return len(self._buffer)

    def __repr__(self) -> str:
        return (
            f"CrowdDataBuffer(size={self.size}, capacity={self.capacity}, "
            f"total_updates={self._total_updates})"
        )
