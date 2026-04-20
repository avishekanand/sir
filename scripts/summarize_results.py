"""
summarize_results.py — Aggregate experiment grid CSVs into a final summary table.

Usage:
    python scripts/summarize_results.py                  # latest run, rich table
    python scripts/summarize_results.py --all-runs       # average across all runs
    python scripts/summarize_results.py --file results/experiment_grid_all_20260222_062844.csv
    python scripts/summarize_results.py --per-dataset    # also show per-dataset breakdown
    python scripts/summarize_results.py --pivot          # dataset-columns × metric sub-columns (IR paper format)
    python scripts/summarize_results.py --markdown       # plain markdown table (no truncation)
    python scripts/summarize_results.py --save           # write summary CSV to results/
"""

import argparse
import datetime
import glob
import os
import sys

import pandas as pd
from rich import box
from rich.console import Console
from rich.table import Table

console = Console()

_REQUIRED_COLUMNS = {"query_id", "ndcg@5", "recall@5", "mrr", "latency_ms", "dataset"}

def _validate_columns(df: pd.DataFrame, path: str):
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        console.print(f"[red]CSV '{path}' is missing required columns: {sorted(missing)}[/red]")
        console.print("[red]Note: summary CSVs produced by --save cannot be reloaded with --file.[/red]")
        sys.exit(1)

GROUP_LABELS = {
    "A": "BM25 Baseline",
    "B": "Budget Ablation (MonoT5)",
    "C": "Estimator Ablation",
    "D": "Feedback Ablation",
    "E": "Ollama vs MonoT5",
    "F": "Full ReformIR Pipeline",
}

METRIC_COLS = ["ndcg@5", "recall@5", "mrr", "latency_ms", "rerank_docs_used", "n_iterations"]


def load_data(args) -> pd.DataFrame:
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")

    if args.file:
        path = args.file
        if not os.path.exists(path):
            console.print(f"[red]File not found: {path}[/red]")
            sys.exit(1)
        console.print(f"[dim]Loading: {path}[/dim]")
        df = pd.read_csv(path)
        _validate_columns(df, path)
        return df

    pattern = os.path.join(results_dir, "experiment_grid_all_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        console.print(f"[red]No experiment_grid_all_*.csv files found in {results_dir}[/red]")
        sys.exit(1)

    if args.all_runs:
        console.print(f"[dim]Merging {len(files)} run(s): {[os.path.basename(f) for f in files]}[/dim]")
        frames = [pd.read_csv(f) for f in files]
        for f, frame in zip(files, frames):
            _validate_columns(frame, f)
        return pd.concat(frames, ignore_index=True)
    else:
        path = files[-1]
        console.print(f"[dim]Loading latest run: {os.path.basename(path)}[/dim]")
        df = pd.read_csv(path)
        _validate_columns(df, path)
        return df


def aggregate(df: pd.DataFrame, groupby: list) -> pd.DataFrame:
    agg = (
        df.groupby(groupby)
        .agg(
            ndcg5=("ndcg@5",           "mean"),
            recall5=("recall@5",       "mean"),
            mrr=("mrr",                "mean"),
            latency=("latency_ms",     "mean"),
            rerank_docs=("rerank_docs_used", "mean"),
            iterations=("n_iterations","mean"),
            n_queries=("query_id",     "count"),
        )
        .reset_index()
    )
    agg["efficiency"] = agg["ndcg5"] / (agg["latency"] / 1000 + 1e-9)
    return agg


def print_summary_table(agg: pd.DataFrame, title: str):
    t = Table(title=title, box=box.SIMPLE_HEAVY, show_lines=False, title_style="bold white")
    t.add_column("Group",       style="dim",        no_wrap=True)
    t.add_column("Config",      style="cyan",       no_wrap=True)
    t.add_column("NDCG@5",      justify="right",    style="bold green")
    t.add_column("Recall@5",    justify="right")
    t.add_column("MRR",         justify="right")
    t.add_column("Latency ms",  justify="right")
    t.add_column("Rerank docs", justify="right")
    t.add_column("Iterations",  justify="right")
    t.add_column("Efficiency",  justify="right",    style="dim")
    t.add_column("N queries",   justify="right",    style="dim")

    prev_group = None
    for _, r in agg.iterrows():
        g = r.get("group", "")
        label = GROUP_LABELS.get(g, g) if g else ""
        group_display = label if g != prev_group else ""
        prev_group = g

        t.add_row(
            group_display,
            str(r["config"]),
            f"{r['ndcg5']:.4f}",
            f"{r['recall5']:.4f}",
            f"{r['mrr']:.4f}",
            f"{r['latency']:.0f}",
            f"{r['rerank_docs']:.1f}",
            f"{r['iterations']:.1f}",
            f"{r['efficiency']:.4f}",
            str(int(r["n_queries"])),
        )

    console.print(t)


def print_per_dataset_table(agg: pd.DataFrame, dataset: str):
    title = f"Per-dataset breakdown — {dataset}"
    print_summary_table(agg, title)


# ── Pivot table: datasets as column groups, configs as rows ──────────────────

PIVOT_METRICS = [
    ("ndcg@5",           "NDCG@5"),
    ("recall@5",         "Rec@5"),
    ("mrr",              "MRR"),
    ("latency_ms",       "Lat(ms)"),
    ("n_iterations",     "Iters"),
]

def print_pivot_table(df: pd.DataFrame):
    """One row per config, one column group per dataset + macro-avg column group."""
    datasets = sorted(df["dataset"].unique())

    # Per-dataset means
    per_ds: dict[str, pd.DataFrame] = {}
    for ds in datasets:
        sub = df[df["dataset"] == ds]
        per_ds[ds] = (
            sub.groupby(["group", "config"])[
                ["ndcg@5", "recall@5", "mrr", "latency_ms", "n_iterations"]
            ]
            .mean()
            .reset_index()
        )

    # Macro-avg across datasets
    macro = (
        df.groupby(["group", "config"])[
            ["ndcg@5", "recall@5", "mrr", "latency_ms", "n_iterations"]
        ]
        .mean()
        .reset_index()
    )

    # Ordered config list (preserving group order)
    config_order = macro[["group", "config"]].values.tolist()

    t = Table(
        title="RAGtune — Pivot: configs × datasets",
        box=box.SIMPLE_HEAVY,
        show_lines=True,
        title_style="bold white",
    )
    t.add_column("Config", style="cyan", no_wrap=True)

    for ds in datasets:
        for _, metric_label in PIVOT_METRICS:
            t.add_column(f"{ds[:6]}\n{metric_label}", justify="right", no_wrap=True)

    for _, metric_label in PIVOT_METRICS:
        t.add_column(f"avg\n{metric_label}", justify="right", style="bold green", no_wrap=True)

    for group, config in config_order:
        row = [config]
        for ds in datasets:
            ds_df = per_ds[ds]
            match = ds_df[(ds_df["group"] == group) & (ds_df["config"] == config)]
            if match.empty:
                row += ["—"] * len(PIVOT_METRICS)
            else:
                r = match.iloc[0]
                fmts = [f"{r['ndcg@5']:.3f}", f"{r['recall@5']:.3f}", f"{r['mrr']:.3f}",
                        f"{r['latency_ms']:.0f}", f"{r['n_iterations']:.1f}"]
                row += fmts
        # macro-avg
        m = macro[(macro["group"] == group) & (macro["config"] == config)].iloc[0]
        row += [f"{m['ndcg@5']:.3f}", f"{m['recall@5']:.3f}", f"{m['mrr']:.3f}",
                f"{m['latency_ms']:.0f}", f"{m['n_iterations']:.1f}"]
        t.add_row(*row)

    console.print(t)


def print_markdown_table(agg: pd.DataFrame):
    """Plain markdown table — configs as rows, metrics as columns. Never truncates."""
    cols = ["group", "config", "ndcg5", "recall5", "mrr", "latency", "rerank_docs", "iterations", "efficiency"]
    header = "| Group | Config | NDCG@5 | Recall@5 | MRR | Latency ms | Rerank docs | Iterations | Efficiency |"
    sep    = "|-------|--------|--------|----------|-----|-----------|-------------|------------|------------|"
    print(header)
    print(sep)
    prev_group = None
    for _, r in agg.iterrows():
        g = r.get("group", "")
        label = GROUP_LABELS.get(g, g) if g else ""
        group_display = label if g != prev_group else ""
        prev_group = g
        print(
            f"| {group_display} | {r['config']} "
            f"| {r['ndcg5']:.4f} | {r['recall5']:.4f} | {r['mrr']:.4f} "
            f"| {r['latency']:.0f} | {r['rerank_docs']:.1f} "
            f"| {r['iterations']:.1f} | {r['efficiency']:.4f} |"
        )


def main():
    parser = argparse.ArgumentParser(description="Summarize RAGtune experiment grid results")
    parser.add_argument("--file",        default=None,  help="Specific CSV file to load")
    parser.add_argument("--all-runs",    action="store_true", help="Merge all available run CSVs")
    parser.add_argument("--per-dataset", action="store_true", help="Also print per-dataset breakdown")
    parser.add_argument("--pivot",       action="store_true", help="Pivot table: dataset columns × metric sub-columns")
    parser.add_argument("--markdown",    action="store_true", help="Print plain markdown table (no truncation)")
    parser.add_argument("--save",        action="store_true", help="Save summary to results/summary_<timestamp>.csv")
    args = parser.parse_args()

    df = load_data(args)
    console.print(f"[dim]Loaded {len(df)} rows across datasets: {sorted(df['dataset'].unique())}[/dim]\n")

    agg_all = aggregate(df, groupby=["group", "config"])

    # ── Main summary: macro-averaged rich table ───────────────────────────────
    print_summary_table(agg_all, "RAGtune Experiment Grid — Macro-averaged across all datasets")

    # ── Pivot table ───────────────────────────────────────────────────────────
    if args.pivot:
        print_pivot_table(df)

    # ── Markdown table ────────────────────────────────────────────────────────
    if args.markdown:
        print_markdown_table(agg_all)

    # ── Per-dataset breakdown ─────────────────────────────────────────────────
    if args.per_dataset:
        for ds in sorted(df["dataset"].unique()):
            sub = df[df["dataset"] == ds]
            agg_ds = aggregate(sub, groupby=["group", "config"])
            print_per_dataset_table(agg_ds, ds)

    # ── Save ──────────────────────────────────────────────────────────────────
    if args.save:
        results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(results_dir, f"summary_{ts}.csv")
        agg_all.to_csv(out_path, index=False)
        console.print(f"\n[bold green]Summary saved → {out_path}[/bold green]")


if __name__ == "__main__":
    main()
