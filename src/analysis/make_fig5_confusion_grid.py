"""Regenerates Figure 10 (fig5_confusion_grid) for the ESWA manuscript.

Six key systems, all recomputed from the current (no-RAG) result files.
Counts with row-normalised proportions in parentheses.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
SCORES = ROOT / "results" / "predictions"
OUT = ROOT / "figures" / "fig5_confusion_grid"

plt.rcParams.update({
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13,
    "xtick.labelsize": 11, "ytick.labelsize": 11,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
})

LABELS = [-1, 0, 1]
NAMES = ["Neg", "Neu", "Pos"]

SYSTEMS = [
    ("Static top-heavy (0.60/0.20/0.20)", "static_top_heavy_voting.csv", "true_label", "final_pred"),
    ("Twitter-RoBERTa (fine-tuned)",      "twitter_roberta.csv",       "sentiment",  "pred_label"),
    ("Static-tier (0.50/0.25/0.25)",      "static_tier_voting.csv",       "true_label", "final_pred"),
    ("Multi-agent (with hints)",          "multi_agent_with_hints.csv",        "true_label", "pred_label"),
    ("RAG over KB A",                     "rag_kb_a.csv",                 "true_label", "pred_label"),
    ("Multi-agent (no hints)",            "multi_agent_no_hints.csv", "true_label", "pred_label"),
]


def main():
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 9.2))

    for ax, (title, fn, tcol, pcol) in zip(axes.ravel(), SYSTEMS):
        df = pd.read_csv(SCORES / fn)
        yt = df[tcol].astype(int)
        yp = df[pcol].astype(int)
        acc = accuracy_score(yt, yp)
        cm = confusion_matrix(yt, yp, labels=LABELS)
        cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)

        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                colour = "white" if cmn[i, j] > 0.5 else "black"
                ax.text(j, i, f"{cm[i, j]}\n({cmn[i, j]:.2f})",
                        ha="center", va="center", fontsize=12, color=colour)

        ax.set_title(f"{title}\nacc={acc:.4f}")
        ax.set_xticks(range(3)); ax.set_xticklabels(NAMES)
        ax.set_yticks(range(3)); ax.set_yticklabels(NAMES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)

        print(f"{title:<36} acc={acc:.4f}")

    fig.tight_layout()
    fig.savefig(f"{OUT}.png", dpi=300)
    fig.savefig(f"{OUT}.pdf")
    plt.close(fig)
    print(f"\nSaved: {OUT}.png / .pdf")


if __name__ == "__main__":
    main()
