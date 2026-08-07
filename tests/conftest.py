"""Shared fixtures and markers.

The tests are correctness gates, not decoration. Each one asserts your
implementation matches a reference, so "am I done with this lecture?" has an
unambiguous answer: pytest passes or it doesn't.

Markers:
    slow    loads the real model (~1.2GB download on first run)
    cuda    needs an NVIDIA GPU -- skipped on the laptop

Run what works locally:
    pytest -m "not cuda"          # everything the M1 can do
    pytest -m "not slow and not cuda"   # fast, no model download
"""

from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: loads the real model")
    config.addinivalue_line("markers", "cuda: requires an NVIDIA GPU")


def pytest_collection_modifyitems(config: pytest.Config, items) -> None:
    """Auto-skip cuda tests when there's no CUDA, with a clear reason."""
    try:
        import torch

        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False

    if has_cuda:
        return
    skip = pytest.mark.skip(reason="needs an NVIDIA GPU (rent a 3090 -- see README)")
    for item in items:
        if "cuda" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def device() -> str:
    from bench.harness import detect_device

    return detect_device()


@pytest.fixture(scope="session")
def model_and_tokenizer():
    """The real Qwen3-0.6B. Session-scoped -- loading is slow, so load once.

    Skips (rather than fails) if the model isn't downloaded yet, so the rest of
    the suite still runs on a fresh clone.
    """
    pytest.importorskip("transformers")
    try:
        from engine.model import load

        return load()
    except NotImplementedError:
        pytest.skip("engine/model.py::load not implemented yet (Lecture 03)")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"could not load model: {exc}")


@pytest.fixture(scope="session")
def reference_greedy(model_and_tokenizer):
    """Ground truth: HuggingFace's own greedy generation.

    Every generation path you write -- cached, batched, paged, speculative --
    must reproduce this EXACTLY. Greedy decoding is deterministic, so any
    divergence is a bug in your implementation, not sampling noise.

    This single fixture is what makes the rest of the course safe: you can
    optimize aggressively because the test tells you the moment you break
    correctness.
    """
    import torch

    model, tok = model_and_tokenizer
    prompt = "The KV cache exists because"
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=24, do_sample=False, temperature=None, top_p=None,
            pad_token_id=tok.eos_token_id,
        )
    return {
        "prompt": prompt,
        "prompt_ids": ids[0].tolist(),
        "output_ids": out[0].tolist(),
        "text": tok.decode(out[0], skip_special_tokens=True),
    }
