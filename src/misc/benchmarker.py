import json
import os
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from time import time

import numpy as np
import torch

# Set PAVESPLAT_SYNC_TIMING to 1 to bracket the encoder timer with
# torch.cuda.synchronize() (encoder-time measurement in the supplement).
_SYNC_TIMING = bool(os.environ.get("PAVESPLAT_SYNC_TIMING"))


class Benchmarker:
    def __init__(self):
        self.execution_times = defaultdict(list)

    @contextmanager
    def time(self, tag: str, num_calls: int = 1):
        try:
            if _SYNC_TIMING and torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time()
            yield
        finally:
            if _SYNC_TIMING and torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time()
            for _ in range(num_calls):
                self.execution_times[tag].append((end_time - start_time) / num_calls)

    def dump(self, path: Path) -> None:
        path.parent.mkdir(exist_ok=True, parents=True)
        with path.open("w") as f:
            json.dump(dict(self.execution_times), f)

    def dump_memory(self, path: Path) -> None:
        path.parent.mkdir(exist_ok=True, parents=True)
        with path.open("w") as f:
            json.dump(torch.cuda.memory_stats()["allocated_bytes.all.peak"], f)

    def summarize(self) -> None:
        for tag, times in self.execution_times.items():
            print(f"{tag}: {len(times)} calls, avg. {np.mean(times)} seconds per call")

    def clear_history(self) -> None:
        self.execution_times = defaultdict(list)
