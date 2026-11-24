#!/usr/bin/env python
"""
Calibrate the cost model's hardware- and corpus-dependent constants.

Emits the n_ctx and ce_ms_per_doc blocks of config/cost_model.yaml. Both are
assumptions that move conclusions, so they are measured and versioned rather
than hardcoded. See specs/spec-cost-model.md section 4.

  python scripts/calibrate_cost_model.py --dataset nfcorpus --device cuda
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

DEFAULT_RERANKERS = [
    "cross-encoder/ms-marco-MiniLM-L-2-v2",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "cross-encoder/ms-marco-MiniLM-L-12-v2",
]
IRDS = {"nfcorpus": "beir/nfcorpus/test", "scifact": "beir/scifact/test"}


def measure_n_ctx(dataset: str, tokenizer_name: str, max_docs: int) -> dict:
    """Effective context length: n_q + l_doc, after max_length truncation."""
    import ir_datasets
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    ds = ir_datasets.load(IRDS[dataset])

    n_q = np.array([
        len(tok(q.text, add_special_tokens=False)["input_ids"])
        for q in ds.queries_iter()
    ])
    l_doc = []
    for i, d in enumerate(ds.docs_iter()):
        if i >= max_docs:
            break
        text = (getattr(d, "title", "") or "") + " " + (getattr(d, "text", "") or "")
        l_doc.append(len(tok(text, add_special_tokens=False)["input_ids"]))
    l_doc = np.array(l_doc)

    cap = int(tok.model_max_length)
    # +3 for [CLS] q [SEP] d [SEP]
    pairs = np.array([min(cap, a + b + 3) for a in n_q for b in l_doc[:400]])
    return {
        "n_ctx": int(round(float(pairs.mean()))),
        "max_length_cap": cap,
        "pct_truncated": round(float((pairs >= cap).mean()) * 100, 1),
        "n_q_mean": round(float(n_q.mean()), 1),
        "l_doc_mean": round(float(l_doc.mean()), 1),
        "n_queries": int(len(n_q)),
        "n_docs_sampled": int(len(l_doc)),
    }


def measure_ms_per_doc(models, dataset, device, n_pairs, warmup, batch_sizes, repeats):
    """
    Wall-clock ms per (query, document) pair, as a {model: {batch_size: ms}} grid.

    Batch size matters: the effective GPU batch equals the tuned
    scheduler_batch_size, and per-doc cost swings ~3.9x over the 1-20 range.

    One FIXED document sample is reused for every cell. Resampling per repeat
    measures a different workload each time and produces spurious
    non-monotonicity (CV ~12% and bs=2 appearing dearer than bs=1).
    """
    import random
    import statistics
    import ir_datasets
    from sentence_transformers import CrossEncoder

    ds = ir_datasets.load(IRDS[dataset])
    query = next(ds.queries_iter()).text
    pool = []
    for i, d in enumerate(ds.docs_iter()):
        if i >= max(1200, n_pairs * 4):
            break
        pool.append((getattr(d, "title", "") or "") + " " + (getattr(d, "text", "") or ""))
    random.seed(42)
    pairs = [(query, d) for d in random.sample(pool, min(n_pairs, len(pool)))]

    out = {}
    for name in models:
        ce = CrossEncoder(name, device=device)
        ce.predict(pairs[:warmup])                       # warm kernels / autotune
        cells = {}
        for bs in batch_sizes:
            reps = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                ce.predict(pairs, batch_size=bs)
                reps.append((time.perf_counter() - t0) * 1000.0 / len(pairs))
            median = statistics.median(reps)
            cv = 100 * statistics.pstdev(reps) / median if median else 0.0
            if cv > 10:
                print(f"  WARNING {name} bs={bs}: CV={cv:.0f}% — unstable, "
                      f"raise --n-pairs or --repeats")
            cells[bs] = round(median, 3)
        out[name] = cells
        del ce
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nfcorpus", choices=sorted(IRDS))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rerankers", default=",".join(DEFAULT_RERANKERS))
    ap.add_argument("--tokenizer", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--max-docs", type=int, default=1500)
    ap.add_argument("--n-pairs", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=64)
    ap.add_argument("--batch-sizes", default="1,2,5,10,20",
                    help="must span the tuned scheduler_batch_size range")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--skip-timing", action="store_true")
    ap.add_argument("--json", action="store_true", help="emit raw JSON too")
    args = ap.parse_args()

    ctx = measure_n_ctx(args.dataset, args.tokenizer, args.max_docs)
    timing = (
        {} if args.skip_timing
        else measure_ms_per_doc(
            args.rerankers.split(","), args.dataset, args.device,
            args.n_pairs, args.warmup,
            [int(b) for b in args.batch_sizes.split(",")], args.repeats,
        )
    )

    if args.json:
        print(json.dumps({"n_ctx": ctx, "ce_ms_per_doc": timing}, indent=2))

    print("\n# --- paste into src/ragtune/config/cost_model.yaml ---")
    print("n_ctx:")
    print(
        f"  {args.dataset}: {ctx['n_ctx']}    # measured: n_q mean "
        f"{ctx['n_q_mean']} + l_doc mean {ctx['l_doc_mean']}, "
        f"{ctx['max_length_cap']} cap, {ctx['pct_truncated']}% clipped"
    )
    if timing:
        print("\nce_ms_per_doc:")
        for name, cells in timing.items():
            print(f"  {name}:")
            for bs, ms in cells.items():
                print(f"    {bs}: {ms}")
        print(f"# provenance: device={args.device}, dataset={args.dataset}, "
              f"fixed {args.n_pairs}-doc workload, median of {args.repeats}, "
              f"warmup={args.warmup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
