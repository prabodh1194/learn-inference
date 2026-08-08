"""M4.2 (Lecture 25) -- open-loop load generation.

The distinction that decides what you can observe:

  CLOSED loop -- N clients each wait for a response before sending again.
                 Load self-limits: if the server slows, offered load drops.
                 Overload is literally unobservable.
  OPEN loop   -- arrivals continue at a fixed rate regardless. If service is
                 slower than arrival, the queue grows without bound.

You want open loop. It's what real traffic does.
"""

from __future__ import annotations


def find_knee(points: list[tuple[float, float, float]]) -> float:
    """Find the offered load where throughput stops rising.

    points: (offered_rate, achieved_throughput, p99_latency)

    Implemented -- it's the analysis, not the measurement. The knee is your
    real capacity: past it, throughput is flat and latency grows without
    bound, so "requests per second" without a latency bound is meaningless.
    """
    if len(points) < 2:
        return float("nan")
    best_rate, best_gain = points[0][0], 0.0
    for (r0, t0, _), (r1, t1, _) in zip(points, points[1:]):
        if r1 == r0:
            continue
        gain = (t1 - t0) / (r1 - r0)          # throughput per unit offered
        if gain > best_gain:
            best_gain, best_rate = gain, r1
        elif gain < best_gain * 0.5:          # returns fell off sharply
            return r0
    return best_rate


def simulate(rate: float, service_time: float, duration: float,
             mode: str = "open", clients: int = 8) -> dict:
    """M4.2. Simulate arrivals and report queue depth plus latency percentiles.

    Must return: queue_depth_initial, queue_depth_final, p50, p99.

    In "open" mode an overloaded run MUST show a growing queue -- that is the
    honest saturation signal, and it appears before the latency numbers do.
    """
    raise NotImplementedError("M4.2")


def run_load_test(base_url: str, workload, rate: float) -> dict:
    """M4.2. Drive a real server (Lecture 24) with Poisson arrivals."""
    raise NotImplementedError("M4.2")
