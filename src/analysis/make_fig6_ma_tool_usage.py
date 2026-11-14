"""Regenerates Figure 8 (fig6_ma_tool_usage) for the ESWA manuscript.

Three-tool configuration (RAG tool removed). Panel (a) shows how often the
supervisor called each classifier; panel (b) shows how many graphs completed
versus errored, with the accuracy on each subset.

Values are computed from
  results/predictions/multi_agent_with_hints.csv          (with hints)
  results/predictions/multi_agent_no_hints.csv (no hints)
"""

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[2]
SCORES = ROOT / "results" / "predictions"
OUT = ROOT / "figures" / "fig6_ma_tool_usage"

# Okabe-Ito colour-blind-safe palette
BLUE      = "#0072B2"   # with hints
AMBER     = "#E69F00"   # no hints
TEAL      = "#009E73"   # graph completed
VERMILION = "#D55E00"   # graph error

plt.rcParams.update({
    "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 11,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})

TOOLS = [
    ("classify_with_twitter_roberta", "Twitter-\nRoBERTa"),
    ("classify_with_roberta_large",   "RoBERTa-\nLarge"),
    ("classify_with_chatgpt",         "gpt-4.1-mini\n(ft)"),
]

RUNS = [
    ("With hints", "multi_agent_with_hints.csv", BLUE),
    ("No hints",   "multi_agent_no_hints.csv", AMBER),
]


def load(fn):
    df = pd.read_csv(SCORES / fn)
    counts = Counter()
    for tc in df["tools_called"].fillna(""):
        # count distinct tools per tweet: a few tweets call the same tool twice,
        # and the panel reports the share of tweets a tool was used on
        for t in {t.strip() for t in str(tc).split(";") if t.strip()}:
            counts[t] += 1
    yt = df["true_label"].astype(int)
    yp = df["pred_label"].astype(int)
    ok = df["status"] == "ok"
    return {
        "n": len(df),
        "counts": counts,
        "n_ok": int(ok.sum()),
        "n_err": int((~ok).sum()),
        "acc_ok": accuracy_score(yt[ok], yp[ok]) if ok.sum() else float("nan"),
        "acc_err": accuracy_score(yt[~ok], yp[~ok]) if (~ok).sum() else float("nan"),
    }


def main():
    data = {name: load(fn) for name, fn, _ in RUNS}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4))

    # ---------------- Panel (a): tool selection
    x = np.arange(len(TOOLS))
    w = 0.36
    for i, (name, _, colour) in enumerate(RUNS):
        d = data[name]
        pct = [100 * d["counts"][k] / d["n"] for k, _ in TOOLS]
        raw = [d["counts"][k] for k, _ in TOOLS]
        bars = ax1.bar(x + (i - 0.5) * w, pct, w, label=name,
                       color=colour, edgecolor="white", linewidth=1.2)
        for b, r, p in zip(bars, raw, pct):
            ax1.text(b.get_x() + b.get_width() / 2, b.get_height() / 2,
                     f"{r}\n({p:.1f}%)", ha="center", va="center",
                     fontsize=11, color="white")

    ax1.set_xticks(x)
    ax1.set_xticklabels([lbl for _, lbl in TOOLS])
    ax1.set_ylabel("% of 200 tweets where tool called")
    ax1.set_ylim(0, 100)
    ax1.set_title("(a) Supervisor tool selection")
    ax1.grid(axis="y", linestyle=":", alpha=0.4)
    ax1.set_axisbelow(True)
    ax1.legend(frameon=False, loc="upper right")

    # ---------------- Panel (b): execution outcome
    labels = [name for name, _, _ in RUNS]
    xb = np.arange(len(labels))
    ok = [data[n]["n_ok"] for n in labels]
    err = [data[n]["n_err"] for n in labels]

    ax2.bar(xb, ok, 0.46, color=TEAL, edgecolor="white", linewidth=1.2,
            label="Graph completed")
    ax2.bar(xb, err, 0.46, bottom=ok, color=VERMILION, edgecolor="white",
            linewidth=1.2, label="Graph error $\\rightarrow$ default neutral")

    for i, name in enumerate(labels):
        d = data[name]
        ax2.text(i, d["n_ok"] / 2, f"{d['n_ok']} completed\nacc {d['acc_ok']:.3f}",
                 ha="center", va="center", color="white", fontsize=11.5)
        if d["n_err"] >= 25:
            # tall enough to hold the label inside the segment
            ax2.text(i, d["n_ok"] + d["n_err"] / 2,
                     f"{d['n_err']} errors\nacc {d['acc_err']:.3f}",
                     ha="center", va="center", color="white", fontsize=11.5)
        else:
            # segment too thin - sit the label above the bar, centred on it
            ax2.text(i, 208, f"{d['n_err']} errors  (acc {d['acc_err']:.3f})",
                     ha="center", va="bottom", color=VERMILION, fontsize=11.5)

    ax2.set_xticks(xb)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Tweets")
    ax2.set_ylim(0, 232)
    ax2.set_yticks(np.arange(0, 201, 50))
    ax2.set_title("(b) Execution outcome and subset accuracy")
    ax2.grid(axis="y", linestyle=":", alpha=0.4)
    ax2.set_axisbelow(True)
    ax2.legend(frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, -0.11), ncol=2)

    fig.tight_layout()
    fig.savefig(f"{OUT}.png", dpi=300)
    fig.savefig(f"{OUT}.pdf")
    plt.close(fig)

    for name in labels:
        d = data[name]
        print(f"{name:<12} ok={d['n_ok']:>3} ({d['acc_ok']:.4f})  "
              f"err={d['n_err']:>3} ({d['acc_err']:.4f})  "
              f"tools={dict(d['counts'])}")
    print(f"\nSaved: {OUT}.png / .pdf")


if __name__ == "__main__":
    main()
