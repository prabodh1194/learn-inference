"""Pull Qwen3-0.6B into ./models (gitignored).

    uv run python scripts/fetch_model.py
"""

from pathlib import Path

MODEL_ID = "Qwen/Qwen3-0.6B"
DEST = Path(__file__).parent.parent / "models"


def main() -> None:
    from huggingface_hub import snapshot_download

    DEST.mkdir(exist_ok=True)
    path = snapshot_download(
        repo_id=MODEL_ID,
        local_dir=DEST / MODEL_ID.split("/")[-1],
        allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*"],
    )
    print(f"-> {path}")
    cfg = Path(path) / "config.json"
    if cfg.exists():
        import json
        c = json.loads(cfg.read_text())
        # These are the numbers M0.4's roofline needs -- verify them there.
        keys = ["num_hidden_layers", "num_attention_heads", "num_key_value_heads",
                "hidden_size", "head_dim", "vocab_size"]
        print("\nconfig (check against notes/00-baseline/m04_roofline.py):")
        for k in keys:
            if k in c:
                print(f"  {k:<26} {c[k]}")


if __name__ == "__main__":
    main()
