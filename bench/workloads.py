"""Request workloads. Each one is shaped to make a specific optimization visible.

The pairing matters -- an optimization measured on the wrong workload looks
like it does nothing:

  uniform          baseline; static batching looks fine here (misleadingly)
  mixed_length     M1.4 continuous batching -- static batching wastes compute
                   waiting on the longest sequence in the batch
  shared_prefix    M1.6 prefix caching -- long common system prompt
  late_divergence  M1.6 the ordering lesson: same tokens, prefix ends at the
                   first difference, so putting novel content early kills reuse
  long_prefill     M1.7 chunked prefill -- long prompts block decode, spiking p99
  code_completion  M1.8 n-gram specdec -- output echoes input, high acceptance
  prose            M1.8 the contrast: n-grams miss, acceptance collapses
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class Request:
    prompt: str
    max_tokens: int = 128
    arrival_offset: float = 0.0  # seconds after run start; 0 = all at once
    tag: str = ""


@dataclass
class Workload:
    name: str
    requests: list[Request]
    note: str = ""
    milestone: str = ""

    def __len__(self) -> int:
        return len(self.requests)

    def __iter__(self) -> Iterator[Request]:
        return iter(self.requests)


# --------------------------------------------------------------------------
# filler text
# --------------------------------------------------------------------------

_WORDS = (
    "system memory bandwidth kernel tensor cache latency throughput batch token "
    "attention decode prefill scheduler block page allocate evict compute stream "
    "warp occupancy pipeline quantize sharding collective overlap fragment"
).split()

SYSTEM_PROMPT = (
    "You are a precise, careful assistant. Answer with technical accuracy. "
    "Prefer concrete numbers over vague claims. When uncertain, say so plainly. "
    "Never fabricate citations or benchmark results. Keep explanations tight "
    "and avoid filler. Use examples when they clarify a mechanism. "
) * 8  # ~400 tokens of shared prefix


def _filler(n_words: int, rng: random.Random) -> str:
    return " ".join(rng.choice(_WORDS) for _ in range(n_words))


# --------------------------------------------------------------------------
# workloads
# --------------------------------------------------------------------------


def uniform(n: int = 16, prompt_words: int = 64, max_tokens: int = 128,
            seed: int = 0) -> Workload:
    rng = random.Random(seed)
    return Workload(
        name="uniform",
        milestone="M0.x",
        note="Equal prompt and output lengths. Baseline only -- static batching "
             "looks deceptively good because there is no padding waste.",
        requests=[
            Request(prompt=_filler(prompt_words, rng), max_tokens=max_tokens)
            for _ in range(n)
        ],
    )


def mixed_length(n: int = 32, seed: int = 0) -> Workload:
    """Wide spread of prompt AND output lengths -- where continuous batching wins."""
    rng = random.Random(seed)
    reqs = []
    for _ in range(n):
        reqs.append(
            Request(
                prompt=_filler(rng.choice([16, 32, 64, 128, 256, 512]), rng),
                max_tokens=rng.choice([8, 16, 64, 128, 256, 512]),
            )
        )
    return Workload(
        name="mixed_length",
        milestone="M1.4",
        note="Static batching stalls: short sequences finish early but their slots "
             "stay occupied until the longest one completes. Compare throughput "
             "AND the fraction of compute spent on padding.",
        requests=reqs,
    )


def shared_prefix(n: int = 24, suffix_words: int = 24, max_tokens: int = 64,
                  seed: int = 0) -> Workload:
    """Long identical system prompt, short unique suffix -- prefix caching's best case."""
    rng = random.Random(seed)
    return Workload(
        name="shared_prefix",
        milestone="M1.6",
        note="~400-token shared system prompt. Measure TTFT before/after prefix "
             "caching; the second request onward should skip nearly all prefill.",
        requests=[
            Request(prompt=SYSTEM_PROMPT + _filler(suffix_words, rng),
                    max_tokens=max_tokens)
            for _ in range(n)
        ],
    )


def late_divergence(n: int = 24, max_tokens: int = 64, seed: int = 0) -> Workload:
    """Same content as shared_prefix, but the unique part comes FIRST.

    Book §5.3.1: the prefix ends at the first non-matching token. Identical
    tokens after a divergence are worthless for reuse. Run this against
    shared_prefix -- near-identical token counts, wildly different hit rates.
    """
    rng = random.Random(seed)
    return Workload(
        name="late_divergence",
        milestone="M1.6",
        note="Novel tokens placed early, killing prefix reuse. The control for "
             "shared_prefix: proves context ORDERING drives cache savings.",
        requests=[
            Request(prompt=_filler(8, rng) + " " + SYSTEM_PROMPT, max_tokens=max_tokens)
            for _ in range(n)
        ],
    )


def long_prefill(n: int = 8, prompt_words: int = 3000, max_tokens: int = 32,
                 n_short: int = 24, seed: int = 0) -> Workload:
    """A few very long prompts mixed with many short ones -- chunked prefill's case."""
    rng = random.Random(seed)
    reqs = [
        Request(prompt=_filler(prompt_words, rng), max_tokens=max_tokens, tag="long")
        for _ in range(n)
    ]
    reqs += [
        Request(prompt=_filler(24, rng), max_tokens=max_tokens, tag="short")
        for _ in range(n_short)
    ]
    rng.shuffle(reqs)
    return Workload(
        name="long_prefill",
        milestone="M1.7",
        note="Long prefills monopolize steps and stall decode for everyone else. "
             "Watch p99 (not mean) on the 'short' requests before/after chunking.",
        requests=reqs,
    )


def code_completion(n: int = 16, max_tokens: int = 128, seed: int = 0) -> Workload:
    """Output heavily echoes input -- n-gram speculation's best case."""
    rng = random.Random(seed)
    template = (
        "def process_batch(requests, scheduler, cache):\n"
        "    scheduled = scheduler.schedule(requests)\n"
        "    for req in scheduled:\n"
        "        block = cache.allocate(req)\n"
        "        req.status = 'running'\n"
        "\n"
        "# Rewrite the function above with error handling, keeping style identical:\n"
    )
    return Workload(
        name="code_completion",
        milestone="M1.8",
        note="Repetitive, predictable syntax. n-gram/prompt-lookup speculation "
             "should show a high acceptance rate here.",
        requests=[
            Request(prompt=template + f"# variant {rng.randrange(1000)}\n",
                    max_tokens=max_tokens)
            for _ in range(n)
        ],
    )


def prose(n: int = 16, max_tokens: int = 128, seed: int = 0) -> Workload:
    """Open-ended generation -- the contrast case where n-gram specdec fails."""
    rng = random.Random(seed)
    topics = [
        "Explain why decode is memory-bound while prefill is compute-bound.",
        "Describe the tradeoff between tensor and expert parallelism.",
        "Why does paging the KV cache reduce fragmentation?",
        "When is speculative decoding a net loss?",
    ]
    return Workload(
        name="prose",
        milestone="M1.8",
        note="Novel output that does not echo the prompt. Acceptance rate should "
             "collapse vs. code_completion -- that contrast IS the lesson.",
        requests=[
            Request(prompt=rng.choice(topics), max_tokens=max_tokens) for _ in range(n)
        ],
    )


def poisson_arrivals(base: Workload, rate: float, seed: int = 0) -> Workload:
    """Stamp Poisson arrival offsets onto a workload (requests/sec).

    Real traffic arrives randomly. Firing everything at t=0 measures a burst,
    not a service -- use this for Phase 4 load testing (M4.2).
    """
    rng = random.Random(seed)
    t = 0.0
    reqs = []
    for r in base.requests:
        t += rng.expovariate(rate)
        reqs.append(
            Request(prompt=r.prompt, max_tokens=r.max_tokens,
                    arrival_offset=t, tag=r.tag)
        )
    return Workload(
        name=f"{base.name}@{rate}rps",
        milestone="M4.2",
        note=f"{base.note} | Poisson arrivals at {rate} req/s.",
        requests=reqs,
    )


ALL = {
    "uniform": uniform,
    "mixed_length": mixed_length,
    "shared_prefix": shared_prefix,
    "late_divergence": late_divergence,
    "long_prefill": long_prefill,
    "code_completion": code_completion,
    "prose": prose,
}
