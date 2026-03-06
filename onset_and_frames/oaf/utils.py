from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    step: int = 0,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "model": model.state_dict(),
        "step": int(step),
    }
    if optimizer is not None:
        obj["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        obj["scheduler"] = scheduler.state_dict()
    if extra is not None:
        obj["extra"] = extra
    torch.save(obj, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    map_location: str = "cpu",
) -> int:
    obj = torch.load(str(path), map_location=map_location)
    model.load_state_dict(obj["model"], strict=True)
    if optimizer is not None and "optimizer" in obj:
        optimizer.load_state_dict(obj["optimizer"])
    if scheduler is not None and "scheduler" in obj:
        scheduler.load_state_dict(obj["scheduler"])
    return int(obj.get("step", 0))


def write_json(path: str | Path, data: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --------------- loss history ---------------

class LossHistory:
    """Accumulates (step, {key: value}) records and saves / plots them."""

    def __init__(self):
        self.records: list[Dict[str, float]] = []

    def append(self, step: int, **kwargs: float) -> None:
        self.records.append({"step": step, **kwargs})

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.records, indent=2), encoding="utf-8")

    def load(self, path: str | Path) -> None:
        p = Path(path)
        if p.exists():
            self.records = json.loads(p.read_text(encoding="utf-8"))

    def plot(self, path: str | Path, title: str = "Training Loss") -> None:
        if not self.records:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return

        keys = [k for k in self.records[0] if k != "step"]
        steps = [r["step"] for r in self.records]

        fig, ax = plt.subplots(figsize=(10, 5))
        for k in keys:
            vals = [r.get(k) for r in self.records]
            if vals[0] is not None:
                ax.plot(steps, vals, label=k, alpha=0.8)
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(path), dpi=150)
        plt.close(fig)
