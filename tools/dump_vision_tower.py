#!/usr/bin/env python3
"""dump_vision_tower.py — extract the Gemma 4 vision tower into a standalone
safetensors file for the geist vision tests/benches.

`src/archs/vision_siglip/vision_encoder.c` loads the SigLIP vision tower and the
multimodal projector by their Hugging Face tensor names. This script copies just
those tensors out of a full Gemma 4 checkpoint into one small file
(`vision_tower.safetensors`) so the vision unit/e2e tests have weights to run
against:

    tests/test_vision_tower_blocks_unit.c   (default: vision_bench/vision_tower.safetensors)
    tests/test_vision_e2e_unit.c

The encoder consumes every tensor under these prefixes (BF16):
    model.vision_tower.*      patch embedder, position table, 16 encoder layers
    model.embed_vision.*      embedding_projection (the multimodal projector)

Usage:
    pip install torch safetensors            # bf16 needs a torch backend
    python3 tools/dump_vision_tower.py SRC -o vision_bench/vision_tower.safetensors

SRC may be:
    * a single *.safetensors file,
    * a checkpoint directory (uses model.safetensors.index.json if present,
      else globs *.safetensors),
    * (with `huggingface_hub` installed) an HF repo id like
      "google/gemma-4-E2B-it" via --from-hub.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PREFIXES = ("model.vision_tower.", "model.embed_vision.")


def wanted(name: str) -> bool:
    return name.startswith(PREFIXES)


def iter_shards(src: Path):
    """Yield safetensors shard paths for a file or directory source."""
    if src.is_file():
        yield src
        return
    index = src / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text())["weight_map"]
        shards = sorted({src / f for n, f in weight_map.items() if wanted(n)})
        if not shards:
            sys.exit(f"dump_vision_tower: no vision tensors in {index}")
        yield from shards
        return
    shards = sorted(src.glob("*.safetensors"))
    if not shards:
        sys.exit(f"dump_vision_tower: no *.safetensors under {src}")
    yield from shards


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="safetensors file, checkpoint dir, or HF repo id (with --from-hub)")
    ap.add_argument("-o", "--out", default="vision_bench/vision_tower.safetensors")
    ap.add_argument("--from-hub", action="store_true",
                    help="treat src as an HF repo id and download it first")
    args = ap.parse_args()

    try:
        import torch  # noqa: F401
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError:
        sys.exit("dump_vision_tower: needs `pip install torch safetensors` "
                 "(bf16 weights require a torch backend).")

    src = Path(args.src)
    if args.from_hub:
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            sys.exit("--from-hub needs `pip install huggingface_hub`")
        src = Path(snapshot_download(args.src, allow_patterns=["*.safetensors", "*.json"]))

    if not src.exists():
        sys.exit(f"dump_vision_tower: source not found: {src}")

    collected: dict = {}
    for shard in iter_shards(src):
        with safe_open(shard, framework="pt") as f:
            for name in f.keys():
                if wanted(name):
                    collected[name] = f.get_tensor(name)
        print(f"  scanned {shard.name}: {len(collected)} vision tensors so far")

    if not collected:
        sys.exit("dump_vision_tower: found 0 matching tensors — is this a Gemma 4 "
                 "checkpoint with a vision tower?")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(collected, str(out))
    print(f"wrote {len(collected)} tensors -> {out}")
    # Sanity: the four tensors the encoder needs by exact name.
    for must in ("model.vision_tower.patch_embedder.input_proj.weight",
                 "model.vision_tower.patch_embedder.position_embedding_table",
                 "model.embed_vision.embedding_projection.weight"):
        flag = "ok" if must in collected else "MISSING"
        print(f"  [{flag}] {must}")


if __name__ == "__main__":
    main()
