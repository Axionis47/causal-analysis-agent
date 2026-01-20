"""Agentic evaluation runner for synthetic datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from app.agents.discovery import _prepare_matrix, _run_pc
from app.agents.treatment import _estimate_psm
from app.agents.validation import _run_refutation
from evals.metrics import ate_bias, precision_recall
from evals.synthetic import generate_simple_causal


def run_once(n: int, seed: int) -> dict:
    df = generate_simple_causal(n=n, seed=seed)
    matrix, names = _prepare_matrix(df)
    graph_payload = _run_pc(matrix, names)
    pred_edges = {(edge["source"], edge["target"]) for edge in graph_payload.get("edges", [])}
    true_edges = {("confounder", "treatment"), ("confounder", "outcome"), ("treatment", "outcome")}
    graph_metrics = precision_recall(true_edges, pred_edges)

    psm = _estimate_psm(df, "treatment", "outcome", ["confounder"])
    if psm is None or psm.get("ate") is None:
        ate = float("nan")
    else:
        ate = float(psm["ate"])
    bias = ate_bias(2.0, ate) if not np.isnan(ate) else float("nan")
    refutation_results = []
    for method in ["placebo_treatment_refuter", "random_common_cause"]:
        result = _run_refutation(df, "treatment", "outcome", ["confounder"], method)
        if result is not None:
            refutation_results.append(result["passed"])
    pass_rate = float(np.mean(refutation_results)) if refutation_results else float("nan")
    return {
        "precision": graph_metrics["precision"],
        "recall": graph_metrics["recall"],
        "f1": graph_metrics["f1"],
        "ate": ate,
        "ate_bias": bias,
        "validation_pass_rate": pass_rate,
    }


def run_eval(runs: int, n: int) -> dict:
    results = [run_once(n=n, seed=seed) for seed in range(42, 42 + runs)]
    ates = [item["ate"] for item in results if not np.isnan(item["ate"])]
    cv = float(np.std(ates) / np.mean(ates)) if ates else float("nan")
    summary = {
        "runs": runs,
        "avg_f1": float(np.mean([item["f1"] for item in results])),
        "avg_ate_bias": float(np.nanmean([item["ate_bias"] for item in results])),
        "avg_validation_pass_rate": float(np.nanmean([item["validation_pass_rate"] for item in results])),
        "consistency_cv": cv,
        "results": results,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    runs = 3 if args.quick else 5
    n = 400 if args.quick else 1200
    summary = run_eval(runs=runs, n=n)

    print(json.dumps(summary, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=2))

    if summary["avg_f1"] < 0.2:
        raise SystemExit("Discovery F1 below threshold")
    if summary["avg_ate_bias"] > 1.0:
        raise SystemExit("ATE bias above threshold")
    if summary["consistency_cv"] > 0.1:
        raise SystemExit("ATE consistency CV above threshold")


if __name__ == "__main__":
    main()
