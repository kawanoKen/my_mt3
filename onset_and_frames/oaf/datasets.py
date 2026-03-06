from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset

from .config import FeatureConfig, LabelConfig, TrainConfig
from .features import LogMelSpec, load_audio_mono, ensure_wave_cache, wave_cache_path
from .labels import LabelTensors, notes_to_labels
from .midi_utils import load_midi_notes


def read_labeled_csv(csv_path: Union[str, Path]) -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            items.append((row["audio_path"], row["midi_path"]))
    return items


def read_split_labeled_csv(
    csv_path: Union[str, Path],
    split: str,
) -> List[Tuple[str, str]]:
    """Read a split-aware labeled CSV (e.g. MAPS_*_scenario.csv)."""
    items: List[Tuple[str, str]] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("split") != split:
                continue
            items.append((row["audio_path"], row["midi_path"]))
    return items


def read_maestro_pairs(
    root: Union[str, Path],
    split: str = "train",
) -> List[Tuple[str, str]]:
    root = Path(root)
    csv_path = root / "maestro-v3.0.0.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"MAESTRO CSV not found: {csv_path}")
    import pandas as _pd
    df = _pd.read_csv(csv_path)
    items: List[Tuple[str, str]] = []
    for _, row in df[df["split"] == split].iterrows():
        audio_path = root / str(row["audio_filename"])
        midi_path = root / str(row["midi_filename"])
        if audio_path.exists() and midi_path.exists():
            items.append((str(audio_path), str(midi_path)))
    return items


def read_unlabeled_csv(csv_path: Union[str, Path]) -> List[str]:
    items: List[str] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            items.append(row["audio_path"])
    return items


def _pad_to_length(x: torch.Tensor, T: int, value: float = 0.0) -> torch.Tensor:
    t = x.shape[-1]
    if t == T:
        return x
    if t > T:
        return x[..., :T]
    pad = torch.full((*x.shape[:-1], T - t), fill_value=value, dtype=x.dtype)
    return torch.cat([x, pad], dim=-1)


class _BasePieceDataset(Dataset):
    """Base dataset with wave-level .npy caching (compatible with my_mt3)."""

    def __init__(
        self,
        feat_cfg: FeatureConfig,
        train_cfg: TrainConfig,
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        self.feat_cfg = feat_cfg
        self.train_cfg = train_cfg
        self.cache_dir = str(cache_dir) if cache_dir is not None else None
        self.featurizer = LogMelSpec(feat_cfg)

    @property
    def segment_frames(self) -> int:
        return int(round(self.train_cfg.segment_seconds * self.feat_cfg.sample_rate / self.feat_cfg.hop_length))

    def check_cache(self, audio_paths: List[str]) -> None:
        """Log how many audio files already have a wave cache."""
        if self.cache_dir is None:
            return
        sr = self.feat_cfg.sample_rate
        hit = sum(
            1 for p in audio_paths
            if wave_cache_path(p, cache_dir=self.cache_dir, sr=sr).exists()
        )
        total = len(audio_paths)
        print(f"[wave cache] {hit}/{total} files cached in {self.cache_dir}")

    def _resolve_audio(self, audio_path: str) -> str:
        """Return audio_path or its .npy wave cache path if cache_dir is set."""
        if str(audio_path).endswith(".npy"):
            return audio_path
        if self.cache_dir is None:
            return audio_path
        return ensure_wave_cache(
            audio_path,
            cache_dir=self.cache_dir,
            sr=self.feat_cfg.sample_rate,
        )

    def _load_mel(self, audio_path: str) -> torch.Tensor:
        """Load audio (from cache if available) and compute log-mel."""
        resolved = self._resolve_audio(audio_path)
        wav, _sr = load_audio_mono(resolved, target_sr=self.feat_cfg.sample_rate)
        with torch.no_grad():
            log_mel = self.featurizer(wav).squeeze(0)  # (F, T)
        return log_mel

    def _random_crop_time(
        self,
        log_mel: torch.Tensor,
        labels: Optional[LabelTensors],
        avoid_note_splits: bool,
    ) -> Tuple[torch.Tensor, Optional[LabelTensors]]:
        T = log_mel.shape[-1]
        seg_T = self.segment_frames
        if T <= seg_T:
            log_mel_c = _pad_to_length(log_mel, seg_T, value=float(log_mel.mean().item()))
            if labels is None:
                return log_mel_c, None
            return log_mel_c, LabelTensors(
                onset=_pad_to_length(labels.onset, seg_T, 0.0),
                frame=_pad_to_length(labels.frame, seg_T, 0.0),
                offset=_pad_to_length(labels.offset, seg_T, 0.0),
                velocity=_pad_to_length(labels.velocity, seg_T, 0.0) if labels.velocity is not None else None,
            )

        start = random.randint(0, T - seg_T)
        if avoid_note_splits and labels is not None:
            for _ in range(50):
                s = random.randint(0, T - seg_T)
                if labels.frame[:, s].sum().item() == 0:
                    start = s
                    break

        end = start + seg_T
        log_mel_c = log_mel[:, start:end]
        if labels is None:
            return log_mel_c, None

        lab_c = LabelTensors(
            onset=labels.onset[:, start:end],
            frame=labels.frame[:, start:end],
            offset=labels.offset[:, start:end],
            velocity=labels.velocity[:, start:end] if labels.velocity is not None else None,
        )
        return log_mel_c, lab_c


class LabeledPianoDataset(_BasePieceDataset):
    def __init__(
        self,
        items: List[Tuple[str, str]],
        feat_cfg: FeatureConfig,
        lab_cfg: LabelConfig,
        train_cfg: TrainConfig,
        cache_dir: Optional[Union[str, Path]] = None,
        compute_velocity: bool = False,
    ):
        super().__init__(feat_cfg, train_cfg, cache_dir)
        self.items = items
        self.lab_cfg = lab_cfg
        self.compute_velocity = compute_velocity
        self.check_cache([wav for wav, _mid in items])

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        audio_path, midi_path = self.items[idx]
        log_mel = self._load_mel(audio_path)
        notes = load_midi_notes(
            midi_path,
            apply_sustain=self.lab_cfg.use_sustain,
            sustain_cc=self.lab_cfg.sustain_cc,
            sustain_threshold=self.lab_cfg.sustain_threshold,
        )
        n_frames = log_mel.shape[-1]
        labs = notes_to_labels(notes, n_frames, self.feat_cfg, self.lab_cfg, compute_velocity=self.compute_velocity)
        log_mel, labs = self._random_crop_time(
            log_mel, labs, avoid_note_splits=self.train_cfg.avoid_note_splits
        )

        out: Dict[str, torch.Tensor] = {
            "log_mel": log_mel,
            "onset": labs.onset,
            "frame": labs.frame,
            "offset": labs.offset,
        }
        if labs.velocity is not None:
            out["velocity"] = labs.velocity
        return out

    def estimate_reference_ratios(self, max_items: Optional[int] = None) -> Dict[str, float]:
        n = len(self.items) if max_items is None else min(len(self.items), max_items)
        ones = {"onset": 0.0, "frame": 0.0, "offset": 0.0}
        zeros = {"onset": 0.0, "frame": 0.0, "offset": 0.0}

        for i in range(n):
            ex = self[i]
            for k in ["onset", "frame", "offset"]:
                lab = ex[k]
                ones[k] += float((lab > 0.5).sum().item())
                zeros[k] += float((lab <= 0.5).sum().item())

        return {k: float(ones[k]) / max(float(zeros[k]), 1.0) for k in ones}


class UnlabeledPianoDataset(_BasePieceDataset):
    def __init__(
        self,
        items: List[str],
        feat_cfg: FeatureConfig,
        train_cfg: TrainConfig,
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        super().__init__(feat_cfg, train_cfg, cache_dir)
        self.items = items
        self.check_cache(items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        audio_path = self.items[idx]
        log_mel = self._load_mel(audio_path)
        log_mel, _ = self._random_crop_time(log_mel, None, avoid_note_splits=False)
        return {"log_mel": log_mel}
