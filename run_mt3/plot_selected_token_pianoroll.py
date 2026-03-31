from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


@dataclass
class NoteEvent:
    pitch: int
    onset: float
    offset: float
    on_tok: int | None = None
    off_tok: int | None = None


def _parse_int_list(text: str) -> list[int]:
    text = text.strip()
    if not text:
        return []
    return [int(x) for x in text.split(",") if x.strip() != ""]


def parse_debug_txt(path: Path) -> dict:
    token_ids: list[int] = []
    selected_token_indices: list[int] = []
    all_notes: list[NoteEvent] = []
    sel_notes: list[NoteEvent] = []
    gt_notes: list[NoteEvent] = []

    re_all = re.compile(
        r"^all\[(?P<idx>\d+)\]\s+pitch=(?P<pitch>-?\d+)\s+onset=(?P<onset>[-0-9.]+)\s+"
        r"offset=(?P<offset>[-0-9.]+)\s+on_tok=(?P<on_tok>-?\d+)\s+off_tok=(?P<off_tok>-?\d+)\s*$"
    )
    re_sel = re.compile(
        r"^sel\[(?P<idx>\d+)\]\s+pitch=(?P<pitch>-?\d+)\s+onset=(?P<onset>[-0-9.]+)\s+"
        r"offset=(?P<offset>[-0-9.]+)\s+on_tok=(?P<on_tok>-?\d+)\s+off_tok=(?P<off_tok>-?\d+)\s*$"
    )
    re_gt = re.compile(
        r"^gt\[(?P<idx>\d+)\]\s+pitch=(?P<pitch>-?\d+)\s+onset=(?P<onset>[-0-9.]+)\s+offset=(?P<offset>[-0-9.]+)\s*$"
    )

    text = path.read_text(encoding="utf-8")
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("token_ids="):
            token_ids = _parse_int_list(line.split("=", 1)[1])
            continue
        if line.startswith("selected_token_indices="):
            selected_token_indices = _parse_int_list(line.split("=", 1)[1])
            continue

        m_all = re_all.match(line)
        if m_all:
            all_notes.append(
                NoteEvent(
                    pitch=int(m_all.group("pitch")),
                    onset=float(m_all.group("onset")),
                    offset=float(m_all.group("offset")),
                    on_tok=int(m_all.group("on_tok")),
                    off_tok=int(m_all.group("off_tok")),
                )
            )
            continue

        m_sel = re_sel.match(line)
        if m_sel:
            sel_notes.append(
                NoteEvent(
                    pitch=int(m_sel.group("pitch")),
                    onset=float(m_sel.group("onset")),
                    offset=float(m_sel.group("offset")),
                    on_tok=int(m_sel.group("on_tok")),
                    off_tok=int(m_sel.group("off_tok")),
                )
            )
            continue

        m_gt = re_gt.match(line)
        if m_gt:
            gt_notes.append(
                NoteEvent(
                    pitch=int(m_gt.group("pitch")),
                    onset=float(m_gt.group("onset")),
                    offset=float(m_gt.group("offset")),
                )
            )

    return {
        "token_ids": token_ids,
        "selected_token_indices": selected_token_indices,
        "all_notes": all_notes,
        "sel_notes": sel_notes,
        "gt_notes": gt_notes,
    }


def _to_arrays(events: list[NoteEvent]) -> tuple[np.ndarray, np.ndarray]:
    if not events:
        return np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=int)
    intervals = np.array([[e.onset, e.offset] for e in events], dtype=float)
    pitches = np.array([e.pitch for e in events], dtype=int)
    order = np.argsort(intervals[:, 0])
    return intervals[order], pitches[order]


def _draw_roll(ax, events: list[NoteEvent], *, color: str, edge: str, alpha: float, label: str) -> None:
    first = True
    for ev in events:
        width = max(1e-4, float(ev.offset) - float(ev.onset))
        ax.add_patch(
            Rectangle(
                (float(ev.onset), float(ev.pitch) - 0.4),
                width,
                0.8,
                facecolor=color,
                edgecolor=edge,
                linewidth=0.5,
                alpha=alpha,
                label=label if first else None,
            )
        )
        first = False


def save_plot(out_png: Path, *, gt_notes: list[NoteEvent], all_notes: list[NoteEvent], sel_notes: list[NoteEvent], title: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.8))

    _draw_roll(ax, gt_notes, color="royalblue", edge="navy", alpha=0.45, label=f"GT ({len(gt_notes)})")
    _draw_roll(ax, all_notes, color="gray", edge="dimgray", alpha=0.35, label=f"Pseudo all ({len(all_notes)})")
    _draw_roll(ax, sel_notes, color="crimson", edge="darkred", alpha=0.85, label=f"Pseudo selected ({len(sel_notes)})")

    _, gt_p = _to_arrays(gt_notes)
    _, all_p = _to_arrays(all_notes)
    _, sel_p = _to_arrays(sel_notes)
    pitch_sets = [arr for arr in (gt_p, all_p, sel_p) if arr.size > 0]
    if pitch_sets:
        pcat = np.concatenate(pitch_sets)
        pmin, pmax = int(pcat.min()) - 2, int(pcat.max()) + 2
    else:
        pmin, pmax = 58, 62

    max_t = 0.0
    for evs in (gt_notes, all_notes, sel_notes):
        if evs:
            max_t = max(max_t, max(float(e.offset) for e in evs))
    max_t = max(max_t, 0.1)

    ax.set_xlim(0.0, max_t)
    ax.set_ylim(pmin, pmax)
    ax.set_xlabel("Time (s, local chunk)")
    ax.set_ylabel("MIDI Pitch")
    ax.set_title(title, fontsize=10)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(
        handles=[
            Line2D([0], [0], color="royalblue", lw=6, alpha=0.45, label=f"GT ({len(gt_notes)})"),
            Line2D([0], [0], color="gray", lw=6, alpha=0.35, label=f"Pseudo all ({len(all_notes)})"),
            Line2D([0], [0], color="crimson", lw=6, alpha=0.85, label=f"Pseudo selected ({len(sel_notes)})"),
        ],
        loc="upper right",
        fontsize=8,
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


def save_token_note_csv(
    out_csv: Path,
    *,
    token_ids: list[int],
    selected_token_indices: list[int],
    all_notes: list[NoteEvent],
    sel_notes: list[NoteEvent],
) -> None:
    sel_set = set(int(i) for i in selected_token_indices)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "selected_token_index",
                "token_id",
                "is_selected",
                "all_on_note_count",
                "all_off_note_count",
                "selected_on_note_count",
                "selected_off_note_count",
                "all_on_note_refs",
                "all_off_note_refs",
                "selected_on_note_refs",
                "selected_off_note_refs",
            ],
        )
        writer.writeheader()

        for tok_idx in sorted(sel_set):
            token_id = token_ids[tok_idx] if 0 <= tok_idx < len(token_ids) else None

            all_on = []
            all_off = []
            for i, n in enumerate(all_notes):
                if n.on_tok == tok_idx:
                    all_on.append(f"all[{i}]")
                if n.off_tok == tok_idx:
                    all_off.append(f"all[{i}]")

            sel_on = []
            sel_off = []
            for i, n in enumerate(sel_notes):
                if n.on_tok == tok_idx:
                    sel_on.append(f"sel[{i}]")
                if n.off_tok == tok_idx:
                    sel_off.append(f"sel[{i}]")

            writer.writerow(
                {
                    "selected_token_index": tok_idx,
                    "token_id": "" if token_id is None else int(token_id),
                    "is_selected": 1,
                    "all_on_note_count": len(all_on),
                    "all_off_note_count": len(all_off),
                    "selected_on_note_count": len(sel_on),
                    "selected_off_note_count": len(sel_off),
                    "all_on_note_refs": ";".join(all_on),
                    "all_off_note_refs": ";".join(all_off),
                    "selected_on_note_refs": ";".join(sel_on),
                    "selected_off_note_refs": ";".join(sel_off),
                }
            )


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot selected-token piano roll from pseudo debug txt")
    ap.add_argument("--debug_txt", type=str, required=True, help="path to pseudo debug sample .txt")
    ap.add_argument("--out_png", type=str, default=None, help="output png path (default: same stem)")
    ap.add_argument("--out_csv", type=str, default=None, help="output csv path (default: same stem)")
    args = ap.parse_args()

    debug_txt = Path(args.debug_txt)
    if not debug_txt.exists():
        raise SystemExit(f"debug txt not found: {debug_txt}")

    if args.out_png is None:
        out_png = debug_txt.with_suffix(".selected_token_roll.png")
    else:
        out_png = Path(args.out_png)
    if args.out_csv is None:
        out_csv = debug_txt.with_suffix(".selected_token_note_map.csv")
    else:
        out_csv = Path(args.out_csv)

    parsed = parse_debug_txt(debug_txt)
    token_ids = parsed["token_ids"]
    selected_token_indices = parsed["selected_token_indices"]
    all_notes = parsed["all_notes"]
    sel_notes = parsed["sel_notes"]
    gt_notes = parsed["gt_notes"]

    if not selected_token_indices:
        print("warning: selected_token_indices is empty.")
    if not all_notes:
        print("warning: pseudo all notes not found.")
    if not gt_notes:
        print("warning: gt notes not found.")

    title = (
        f"{debug_txt.name}\n"
        f"selected_tokens={len(selected_token_indices)}  all_notes={len(all_notes)}  selected_notes={len(sel_notes)}  gt_notes={len(gt_notes)}"
    )
    save_plot(out_png, gt_notes=gt_notes, all_notes=all_notes, sel_notes=sel_notes, title=title)
    save_token_note_csv(
        out_csv,
        token_ids=token_ids,
        selected_token_indices=selected_token_indices,
        all_notes=all_notes,
        sel_notes=sel_notes,
    )

    print(f"saved png -> {out_png}")
    print(f"saved csv -> {out_csv}")


if __name__ == "__main__":
    main()
