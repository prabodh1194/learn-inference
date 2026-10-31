"""M0.4 -- roofline and arithmetic intensity by hand.

    uv run python notes/00-baseline/m04_roofline.py

Runs the Lecture 02 roofline arithmetic (no GPU or model download needed) and
saves the roofline chart to bench/results/m04-roofline.png.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bench.plot import roofline  # noqa: E402
from book.code.roofline import (  # noqa: E402
    DEVICES,
    ModelDims,
    decode_step_intensity,
    main as run_roofline_tables,
    prefill_intensity,
)


def main() -> None:
    run_roofline_tables()
    m, dev = ModelDims(), DEVICES[3]
    dec_i = decode_step_intensity(m, seq_len=2048)
    pre_i = prefill_intensity(m, n_tokens=2048)
    points = {
        "decode (B=1)": (dec_i, min(dev.bandwidth * dec_i, dev.peak_flops)),
        "prefill (B=1, N=2048)": (pre_i, min(dev.bandwidth * pre_i, dev.peak_flops)),
    }
    roofline(dev.peak_flops, dev.bandwidth, points, "m04-roofline.png", f"({dev.name})")


if __name__ == "__main__":
    main()
