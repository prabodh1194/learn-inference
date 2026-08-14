"""Lecture 07/08 -- what static batching wastes, made visible.

Replays the exact slot-counting simulation from `batching_waste.py` as a
two-panel animation: the same requests, same seed, same colors.

    uv run python book/code/batch_animation.py            # renders assets/batch-waste.gif
    uv run python book/code/batch_animation.py --full     # the real 32-request run (asserts 61.0%)
    uv run python book/code/batch_animation.py --terminal # ASCII replay in the shell

The point: the 61% figure is not abstract. Top panel, a finished sequence's
slot sits dead until its batch's longest member ends; bottom panel, the same
requests stream without idle slots. The per-step trace below mirrors the
demo's counting rules rule-for-rule, and the final assertion checks the two
aggregates agree with `batching_waste.py`'s numbers.

Default run scales the demo's lengths down by 8 (outputs from
{8,16,32,64,128} instead of {64,...,512}) so the whole run fits on screen;
`--full` uses the untouched workload and must reproduce 61.0% / 0.0% / 2.57x.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from typing import Callable

from PIL import Image, ImageDraw, ImageFont

from bench.workloads import mixed_length
from book.code.batching_waste import (
    _lengths,
    continuous_batch_cost,
    static_batch_cost,
)

# ---------------------------------------------------------------------------
# The simulation (rule-for-rule the same as batching_waste.py, plus a per-step
# trace so we can draw it)
# ---------------------------------------------------------------------------


def trace_static(reqs: list[tuple[int, int]], batch_size: int) -> list[list]:
    """One frame per decode step. Frame is a list of (req_index, status),
    status "run" while the request generates, "dead" after it finished but
    its slot is still held until the batch's longest member ends.
    """
    frames: list[list] = []
    for i in range(0, len(reqs), batch_size):
        chunk = reqs[i:i + batch_size]
        longest = max(o for _, o in chunk)
        for step in range(longest):
            frames.append([
                (i + j, "run" if step < o else "dead")
                for j, (_, o) in enumerate(chunk)
            ])
    return frames


def trace_continuous(reqs: list[tuple[int, int]], n_slots: int) -> list[list]:
    """One frame per decode step. Frame is a list of req_index or None
    (empty slot). A slot that frees admits the next waiting request at the
    following step; no slot ever idles while the queue is non-empty.
    """
    waiting = list(range(len(reqs)))
    slots: list[int | None] = [None] * n_slots
    done: dict[int, int] = {}
    frames: list[list] = []

    for k in range(n_slots):
        if waiting:
            slots[k] = waiting.pop(0)

    while any(s is not None for s in slots):
        for k, s in enumerate(slots):
            if s is not None:
                done[s] = done.get(s, 0) + 1
        frames.append(list(slots))
        for k, s in enumerate(slots):
            if s is not None and done[s] >= reqs[s][1]:
                slots[k] = None
        for k, s in enumerate(slots):
            if s is None and waiting:
                slots[k] = waiting.pop(0)
    return frames


def _check_trace(reqs, batch_size, n_slots) -> None:
    """The picture must not disagree with the demo's numbers."""
    s = static_batch_cost(reqs, batch_size)
    c = continuous_batch_cost(reqs)

    st = trace_static(reqs, batch_size)
    ct = trace_continuous(reqs, n_slots)
    st_slots = len(st) * batch_size
    st_useful = sum(1 for f in st for (_, st_) in f if st_ == "run")
    ct_slots = sum(1 for f in ct for x in f if x is not None)

    assert st_slots == s["decode_slots"], f"{st_slots} vs {s['decode_slots']}"
    assert st_useful == s["decode_useful"], f"{st_useful} vs {s['decode_useful']}"
    assert ct_slots == c["decode_slots"], f"{ct_slots} vs {c['decode_slots']}"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

PALETTE = [
    (231, 76, 60), (46, 204, 113), (52, 152, 219), (241, 196, 15),
    (155, 89, 182), (26, 188, 156), (230, 126, 34), (149, 165, 166),
    (255, 99, 132), (72, 201, 176), (84, 160, 255), (255, 205, 86),
    (247, 118, 142), (94, 187, 124), (133, 133, 253), (140, 190, 238),
    (240, 144, 53), (120, 196, 100),
]


def _color(i: int) -> tuple[int, int, int]:
    return PALETTE[i % len(PALETTE)]


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render(st_frames, ct_frames, reqs, batch_size, n_slots, fps: int,
           out_path: str) -> None:
    n_steps = max(len(st_frames), len(ct_frames))
    cell_h, gap = 15, 1
    grid_h = batch_size * (cell_h + gap)
    margin, title_h, counter_h, legend_h = 14, 22, 18, 26
    cell_w = max(3, min(14, 900 // n_steps))

    width = margin * 2 + n_steps * (cell_w + gap)
    panel_h = title_h + counter_h + grid_h
    height = margin * 2 + panel_h * 2 + legend_h + 10
    canvas = Image.new("RGB", (width, height), (17, 17, 17))
    draw = ImageDraw.Draw(canvas)
    small, mini = _font(13), _font(8)

    def draw_panel(top: int, title: str, dead_note: str, frames, mode: str):
        draw.text((margin, top), title, font=small, fill=(220, 220, 220))
        x0, y0 = margin, top + title_h + counter_h
        for step, frame in enumerate(frames):
            x = x0 + step * (cell_w + gap)
            for row, cell in enumerate(frame):
                y = y0 + row * (cell_h + gap)
                if cell is None:
                    fill = (30, 30, 30)
                elif mode == "static" and cell[1] == "dead":
                    fill = (62, 56, 56)
                else:
                    fill = _color(cell[0] if isinstance(cell, tuple) else cell)
                draw.rectangle((x, y, x + cell_w, y + cell_h), fill=fill)

        # waste counter
        if mode == "static":
            slots = len(frames) * batch_size
            useful = sum(1 for f in frames for (_, st_) in f if st_ == "run")
            pct = 100 * (1 - useful / slots) if slots else 0.0
            text = f"wasted slot-steps: {slots - useful:,} of {slots:,}  ({pct:.1f}%)"
            draw.text((x0, top + title_h), text, font=small,
                      fill=(255, 120, 120))
        else:
            slots = sum(1 for f in frames for x in f if x is not None)
            useful = sum(o for _, o in reqs)
            pct = 100 * (1 - useful / slots) if slots else 0.0
            text = f"idle slot-steps: {slots - useful:,} of {slots:,}  ({pct:.1f}%)"
            draw.text((x0, top + title_h), text, font=small,
                      fill=(140, 255, 170))
        if dead_note:
            draw.text((x0, top + title_h + 4), dead_note, font=mini,
                      fill=(130, 130, 130))

    draw_panel(margin, "STATIC batching: slots held until the batch's longest ends",
               "dead slots = finished requests still occupying the batch",
               st_frames, "static")
    draw_panel(margin + panel_h, "CONTINUOUS batching: finished slots refill next step",
               "the same requests, same colors, none of the waste",
               ct_frames, "cont")

    # legend: request index -> output length, order of arrival
    ly = margin * 2 + panel_h * 2
    x = margin
    for i, (_, out) in enumerate(reqs):
        draw.rectangle((x, ly, x + 12, ly + 12), fill=_color(i))
        draw.text((x + 15, ly - 1), str(out), font=mini, fill=(200, 200, 200))
        x += 15 + 12 + 3 + len(str(out)) * 6
        if x > width - 60:
            x = margin
            ly += 16

    frames: list[Image.Image] = []
    for step in range(n_steps):
        frame = canvas.copy()
        fd = ImageDraw.Draw(frame)
        px = margin + step * (cell_w + gap)
        fd.line((px, 6, px, height - 6), fill=(255, 200, 60))
        frames.append(frame)
    frames.extend([frames[-1]] * 12)  # hold the end state

    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=int(1000 / fps), loop=0, optimize=True)


def terminal(st_frames, ct_frames, reqs, batch_size, fps: int) -> None:
    n_steps = max(len(st_frames), len(ct_frames))
    print(f"\nStatic holding slots  |  Continuous refilling ({len(reqs)} requests, "
          f"{batch_size} slots)\n")
    for step in range(n_steps):
        st = st_frames[step] if step < len(st_frames) else st_frames[-1]
        ct = ct_frames[step] if step < len(ct_frames) else [None] * batch_size
        st_line = "".join("#" if c[1] == "run" else "." for c in st)
        ct_line = "".join("#" if x is not None else " " for x in ct)
        print(f"{st_line}  |  {ct_line}")
        time.sleep(1 / fps)
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full", action="store_true",
                    help="the real 32-request mixed_length workload")
    ap.add_argument("--terminal", action="store_true",
                    help="ASCII replay instead of the GIF")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--out", default="book/assets/batch-waste.gif")
    args = ap.parse_args()

    random.seed(0)
    if args.full:
        reqs = _lengths(mixed_length(n=32, seed=0))
    else:
        # Same workload, lengths scaled down by 8 so the whole run fits on
        # screen: outputs {1,2,8,16,32,64} instead of {8,...,512}.
        reqs = [(p, max(1, o // 8)) for p, o in _lengths(mixed_length(n=16, seed=0))]
    batch_size, n_slots = 8, 8

    _check_trace(reqs, batch_size, n_slots)

    st = trace_static(reqs, batch_size)
    ct = trace_continuous(reqs, n_slots)

    if args.full:
        s = static_batch_cost(reqs, batch_size)
        c = continuous_batch_cost(reqs)
        waste = 100 * (1 - s["decode_useful"] / s["decode_slots"])
        speedup = s["decode_slots"] / c["decode_slots"]
        assert abs(waste - 61.0) < 0.1, f"expected the demo's 61.0%, got {waste:.1f}%"
        assert abs(speedup - 2.57) < 0.01
        print(f"full workload ok: decode waste {waste:.1f}% (demo says 61.0%), "
              f"speedup {speedup:.2f}x (demo says 2.57x)")
        return  # rendering 1100 frames is slow and pointless; the check is the point

    if args.terminal:
        terminal(st, ct, reqs, batch_size, args.fps)
        return

    render(st, ct, reqs, batch_size, n_slots, args.fps, args.out)
    print(f"wrote {args.out} "
          f"({max(len(st), len(ct))} steps, "
          f"{sum(o for _, o in reqs):,} useful tokens)")


if __name__ == "__main__":
    main()