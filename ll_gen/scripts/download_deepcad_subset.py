"""Materialize a local DeepCAD subset from the public ``palapav/DeepCAD-DSL``.

Streams the DeepCAD-DSL dataset from the HuggingFace Hub, flattens each DSL
target into the ``{"sequence": [{"type", "params"}]}`` schema that
``DeepCADDataset`` reads locally, and writes one JSON file per sample under
``<out>/<split>/``.  This gives a reproducible, offline subset for warm-start
training (SPEC-1 M3, T3.1) without committing data into git.

Usage::

    python -m scripts.download_deepcad_subset --split train --n 2000 \\
        --out data/deepcad_dsl
    python -m scripts.download_deepcad_subset --split validation --n 200 \\
        --out data/deepcad_dsl --local-split val

Re-running is idempotent for a fixed ``--n`` and split (streaming order is
deterministic), so the recorded sample count is reproducible.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from ll_gen.datasets._deepcad_dsl import parse_deepcad_dsl

_log = logging.getLogger(__name__)

_HF_DATASET = "palapav/DeepCAD-DSL"


def download_subset(
    n: int,
    hf_split: str,
    out_dir: Path,
    local_split: str,
    min_commands: int = 2,
) -> int:
    """Stream ``n`` samples and write parsed sequences as per-sample JSON.

    Args:
        n: Number of samples to write.
        hf_split: Hub split to stream ("train", "validation", "test").
        out_dir: Root output directory.
        local_split: Subdirectory name under ``out_dir`` (e.g. "train", "val").
        min_commands: Skip samples whose parsed sequence has fewer commands than
            this (drops empty/degenerate DSL strings).

    Returns:
        Number of JSON files written.
    """
    from datasets import load_dataset

    split_dir = out_dir / local_split
    split_dir.mkdir(parents=True, exist_ok=True)

    stream = load_dataset(_HF_DATASET, split=hf_split, streaming=True)

    written = 0
    seen = 0
    for sample in stream:
        if written >= n:
            break
        seen += 1
        dsl = sample.get("target") or sample.get("output") or ""
        sequence = parse_deepcad_dsl(dsl)
        # parse_deepcad_dsl appends an EOS, so a real sketch has >= min+1 cmds.
        if len(sequence) < min_commands + 1:
            continue
        uid = sample.get("sample_id") or sample.get("uid") or f"sample_{written:06d}"
        record = {"uid": uid, "sequence": sequence}
        out_path = split_dir / f"{written:06d}.json"
        out_path.write_text(json.dumps(record))
        written += 1

    _log.info("Wrote %d/%d samples (scanned %d) to %s", written, n, seen, split_dir)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=2000, help="Samples to write.")
    parser.add_argument(
        "--split", default="train", help="Hub split (train/validation/test)."
    )
    parser.add_argument(
        "--local-split",
        default=None,
        help="Local subdir name (defaults to the hub split name).",
    )
    parser.add_argument("--out", default="data/deepcad_dsl", help="Output root.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    count = download_subset(
        n=args.n,
        hf_split=args.split,
        out_dir=Path(args.out),
        local_split=args.local_split or args.split,
    )
    print(json.dumps({"dataset": _HF_DATASET, "split": args.split, "written": count}))


if __name__ == "__main__":
    main()
