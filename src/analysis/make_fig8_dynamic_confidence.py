"""Regenerates Figure 7 (fig8_dynamic_confidence) for the ESWA manuscript.

Panel (a) now carries a third bar - the saturation rate, i.e. the fraction of
tweets on which a component reports confidence >= 0.99. This is the quantity
that actually decides the per-tweet ranking in the dynamic scheme, and without
it panel (a) appears to contradict panel (b): RoBERTa-Large posts the highest
mean confidence yet wins the 0.50 weight on only 9% of tweets.

All values are computed from results/predictions/dynamic_confidence_voting.csv.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "predictions" / "dynamic_confidence_voting.csv"
OUT = ROOT / "figures" / "fig8_dynamic_confidence"

SAT_THRESHOLD = 0.99

# Wong / Okabe-Ito colour-blind-safe palette
PAL = {
    "orange": "#E69F00", "skyblue": "#56B4E9", "green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "vermilion": "#D55E00",
    "purple": "#CC79A7", "grey": "#999999", "black": "#000000",
}

plt.rcParams.update({
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})

COMPONENTS = [
    ("tw", "Twitter-\nRoBERTa"),
    ("rl", "RoBERTa-\nLarge"),
    ("gp", "gpt-4.1-mini\n(ft)"),
]
TOP_MODEL_KEY = {"tw": "twitter_roberta", "rl": "roberta_large", "gp": "chatgpt_ft"}


def load():
    df = pd.read_csv(RESULTS)
    y = df["true_label"].astype(int)
    stats = {}
    for key, _ in COMPONENTS:
        conf = df[f"{key}_conf"].values
        stats[key] = {
            "acc": accuracy_score(y, df[f"{key}_pred"].astype(int)),
            "mean_conf": float(conf.mean()),
            "saturation": float((conf >= SAT_THRESHOLD).mean()),
            "wins": int((df["top_model"] == TOP_MODEL_KEY[key]).sum()),
        }
    return df, stats


def label_inside(ax, bars, fmt="{:.3f}", colour="white", size=10):
    """Write each bar's value at the vertical midpoint of the bar itself."""
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h / 2, fmt.format(h),
                ha="center", va="center", fontsize=size, color=colour)


def main():
    df, stats = load()
    n = len(df)
    labels = [lbl for _, lbl in COMPONENTS]
    keys = [k for k, _ in COMPONENTS]

    acc = [stats[k]["acc"] for k in keys]
    mconf = [stats[k]["mean_conf"] for k in keys]
    sat = [stats[k]["saturation"] for k in keys]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.8))

    # ---------------- Panel (a): accuracy vs mean confidence vs saturation
    x = np.arange(len(labels))
    w = 0.26
    b1 = ax1.bar(x - w, acc, w, label="Component accuracy",
                 color=PAL["blue"], edgecolor="black", linewidth=0.5)
    b2 = ax1.bar(x, mconf, w, label="Mean reported confidence",
                 color=PAL["yellow"], edgecolor="black", linewidth=0.5)
    b3 = ax1.bar(x + w, sat, w, label=f"Saturation rate ($\\geq$ {SAT_THRESHOLD:.2f})",
                 color=PAL["vermilion"], edgecolor="black", linewidth=0.5)

    label_inside(ax1, b1, "{:.3f}", "white")
    label_inside(ax1, b2, "{:.3f}", "black")
    label_inside(ax1, b3, "{:.1%}", "white")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0.0, 1.02)
    ax1.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax1.set_ylabel("Proportion")
    ax1.set_title("(a) Accuracy, mean confidence, and saturation")
    ax1.grid(axis="y", linestyle=":", alpha=0.5)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3,
               frameon=False, fontsize=9, columnspacing=1.2, handlelength=1.4)

    # ---------------- Panel (b): who receives the 0.50 weight
    order = sorted(keys, key=lambda k: -stats[k]["wins"])
    wins = [stats[k]["wins"] for k in order]
    wlabels = [dict(COMPONENTS)[k] for k in order]
    colors = {"gp": PAL["vermilion"], "rl": PAL["grey"], "tw": PAL["blue"]}
    bars = ax2.bar(np.arange(len(order)), wins, 0.6,
                   color=[colors[k] for k in order],
                   edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, wins):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 3,
                 f"{v} ({100*v/n:.0f}%)", ha="center", va="bottom", fontsize=10)

    ax2.set_xticks(np.arange(len(order)))
    ax2.set_xticklabels(wlabels)
    ax2.set_ylim(0, max(wins) * 1.22)
    ax2.set_ylabel("Tweets ranked most confident\n(receives the 0.50 weight)")
    ax2.set_title("(b) Which model receives the 0.50 weight")
    ax2.grid(axis="y", linestyle=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(f"{OUT}.png", dpi=300)
    fig.savefig(f"{OUT}.pdf")
    plt.close(fig)

    print(f"n = {n}")
    for k, lbl in COMPONENTS:
        s = stats[k]
        print(f"  {lbl.replace(chr(10),' '):<24} acc={s['acc']:.4f}  "
              f"mean_conf={s['mean_conf']:.4f}  sat={s['saturation']*100:.1f}%  "
              f"top-rank wins={s['wins']} ({100*s['wins']/n:.1f}%)")
    print(f"\nSaved: {OUT}.png / .pdf")


if __name__ == "__main__":
    main()
