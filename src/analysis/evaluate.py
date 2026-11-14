"""Reproduce the paper's main results table and paired significance tests
from the saved per-tweet predictions in results/predictions/.

Run from the repository root:
    python src/analysis/evaluate.py

Writes results/main_results.csv and results/significance_tests.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[2]
PRED = ROOT / "results" / "predictions"
OUT = ROOT / "results"

# system name -> (file, prediction column)
SYSTEMS = {
    "Static top-heavy voting (0.60/0.20/0.20)": ("static_top_heavy_voting.csv", "final_pred"),
    "Twitter-RoBERTa (fine-tuned)": ("twitter_roberta.csv", "pred_label"),
    "Static-tier voting (0.50/0.25/0.25)": ("static_tier_voting.csv", "final_pred"),
    "Multi-agent supervisor (with hints)": ("multi_agent_with_hints.csv", "pred_label"),
    "Dynamic confidence-ranked voting": ("dynamic_confidence_voting.csv", "final_pred"),
    "gpt-4.1-mini (fine-tuned, no RAG)": ("gpt41_mini.csv", "pred_label"),
    "RoBERTa-Large (fine-tuned)": ("roberta_large.csv", "pred_label"),
    "RAG over KB A (hand-authored)": ("rag_kb_a.csv", "pred_label"),
    "RAG over KB B (corpus-derived)": ("rag_kb_b.csv", "pred_label"),
    "Multi-agent supervisor (no hints)": ("multi_agent_no_hints.csv", "pred_label"),
}

COMPARISONS = [
    ("Static top-heavy voting (0.60/0.20/0.20)", "Twitter-RoBERTa (fine-tuned)"),
    ("Static top-heavy voting (0.60/0.20/0.20)", "Static-tier voting (0.50/0.25/0.25)"),
    ("Static top-heavy voting (0.60/0.20/0.20)", "Dynamic confidence-ranked voting"),
    ("Static top-heavy voting (0.60/0.20/0.20)", "Multi-agent supervisor (with hints)"),
    ("Static top-heavy voting (0.60/0.20/0.20)", "RAG over KB A (hand-authored)"),
    ("Static-tier voting (0.50/0.25/0.25)", "Twitter-RoBERTa (fine-tuned)"),
    ("Multi-agent supervisor (with hints)", "Twitter-RoBERTa (fine-tuned)"),
    ("Multi-agent supervisor (with hints)", "Multi-agent supervisor (no hints)"),
    ("gpt-4.1-mini (fine-tuned, no RAG)", "RAG over KB A (hand-authored)"),
    ("gpt-4.1-mini (fine-tuned, no RAG)", "RAG over KB B (corpus-derived)"),
    ("RAG over KB A (hand-authored)", "RAG over KB B (corpus-derived)"),
]

N_BOOT = 10_000
SEED = 42


def load_predictions():
    test = pd.read_csv(ROOT / "data" / "test.csv")
    y_true = test["sentiment"].astype(int).to_numpy()
    preds = {}
    for name, (fn, col) in SYSTEMS.items():
        df = pd.read_csv(PRED / fn)
        assert len(df) == len(test), f"{fn}: expected {len(test)} rows, got {len(df)}"
        if "tweet" in df.columns:
            assert (df["tweet"].to_numpy() == test["tweet"].to_numpy()).all(), f"{fn}: row order differs from test.csv"
        preds[name] = df[col].astype(int).to_numpy()
    return y_true, preds


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * binom.cdf(min(b, c), n, 0.5))


def bootstrap_ci(correct_a, correct_b):
    # Re-seeded per comparison so each CI is reproducible on its own.
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(correct_a), size=(N_BOOT, len(correct_a)))
    diffs = correct_a[idx].mean(axis=1) - correct_b[idx].mean(axis=1)
    return np.percentile(diffs, [2.5, 97.5])


def main():
    y_true, preds = load_predictions()

    rows = []
    for name, y_pred in preds.items():
        per_class = f1_score(y_true, y_pred, labels=[-1, 0, 1], average=None)
        rows.append({
            "system": name,
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro"),
            "neg_f1": per_class[0], "neu_f1": per_class[1], "pos_f1": per_class[2],
        })
    main_df = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    main_df.to_csv(OUT / "main_results.csv", index=False, float_format="%.4f")
    print("Main results (200-tweet test set)\n")
    print(main_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    sig = []
    for a, b_name in COMPARISONS:
        ca = preds[a] == y_true
        cb = preds[b_name] == y_true
        b = int((ca & ~cb).sum())
        c = int((~ca & cb).sum())
        lo, hi = bootstrap_ci(ca.astype(float), cb.astype(float))
        p = mcnemar_exact(b, c)
        sig.append({
            "A": a, "B": b_name, "acc_A": ca.mean(), "acc_B": cb.mean(),
            "b": b, "c": c, "mcnemar_p": p, "ci_low": lo, "ci_high": hi,
            "significant": p < 0.05,
        })
    sig_df = pd.DataFrame(sig)
    sig_df.to_csv(OUT / "significance_tests.csv", index=False, float_format="%.4f")
    print("\nPaired significance tests (exact McNemar, bootstrap 95% CI)\n")
    print(sig_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
