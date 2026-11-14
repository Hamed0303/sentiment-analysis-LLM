"""Dynamic confidence-ranked weighted voting, WITHOUT RAG.

Identical to dynamic_weighted_pipeline.py except that the third voter is plain
fine-tuned gpt-4.1-mini rather than the RAG-augmented variant, so the voter set
matches the two static schemes exactly.

Voters (all three report a real confidence):
  1. Twitter-RoBERTa           softmax-max
  2. RoBERTa-Large             softmax-max
  3. gpt-4.1-mini (fine-tuned) exp(first-token logprob)

Per tweet the three are ranked by reported confidence; the most confident
receives w=0.50 and the other two w=0.25 each. Aggregation is

    score(c) = sum_m ( w_m * conf_m * 1[pred_m = c] )

Output: results/predictions/dynamic_confidence_voting.csv
"""

import argparse
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from openai import OpenAI
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FT_CHAT_MODEL = os.getenv("FINETUNED_MODEL_HF")
HF_TOKEN = os.getenv("Hugging_Face") or os.getenv("HF_TOKEN")

TWITTER_MODEL = "Hamed0303/twitter-roberta-hf-finetuned"
ROBERTA_MODEL = "Hamed0303/roberta-large-hf-finetuned"

MAX_WORKERS = 8

WEIGHT_TOP = 0.50
WEIGHT_MID = 0.25
WEIGHT_BOT = 0.25

if not FT_CHAT_MODEL:
    raise RuntimeError("FINETUNED_MODEL_HF not set in .env")
if HF_TOKEN:
    login(token=HF_TOKEN)

client = OpenAI(api_key=OPENAI_API_KEY)
device = "cuda" if torch.cuda.is_available() else "cpu"

ROOT = Path(__file__).resolve().parents[2]
TEST_FILE = ROOT / "data" / "test.csv"
RESULTS_FILE = ROOT / "results" / "predictions" / "dynamic_confidence_voting.csv"

REVERSE_MAP = {0: -1, 1: 0, 2: 1}

print(f"[init] Device: {device}")
print(f"[init] Fine-tuned classifier: {FT_CHAT_MODEL}")
print(f"[init] Dynamic weights: top={WEIGHT_TOP}, others={WEIGHT_MID}/{WEIGHT_BOT}  (no RAG)")

print("[init] Loading Twitter-RoBERTa...")
TW_TOK = AutoTokenizer.from_pretrained(TWITTER_MODEL, token=HF_TOKEN)
TW_MODEL = AutoModelForSequenceClassification.from_pretrained(TWITTER_MODEL, token=HF_TOKEN).to(device)
TW_MODEL.eval()

print("[init] Loading RoBERTa-Large...")
RL_TOK = AutoTokenizer.from_pretrained(ROBERTA_MODEL, token=HF_TOKEN)
RL_MODEL = AutoModelForSequenceClassification.from_pretrained(ROBERTA_MODEL, token=HF_TOKEN).to(device)
RL_MODEL.eval()


def encoder_predict_one(tokenizer, model, tweet):
    inputs = tokenizer(tweet, return_tensors="pt", truncation=True,
                       max_length=128, padding=True).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    return REVERSE_MAP[probs.argmax().item()], float(probs.max().item())


SYSTEM_PROMPT_BASE = (
    "You are a sentiment classifier. Classify tweets as -1 (negative), "
    "0 (neutral), or 1 (positive). Reply with only the number."
)


def parse_label(raw):
    if raw is None:
        return 0
    s = str(raw).strip().lower().replace("−", "-").splitlines()
    s = s[0] if s else ""
    if re.fullmatch(r"-1|0|1", s):
        return int(s)
    m = re.search(r"(?<![0-9])-1(?![0-9])|(?<![0-9.])0(?![0-9])|(?<![0-9.])1(?![0-9])", s)
    if m:
        return int(m.group(0))
    if "negative" in s: return -1
    if "positive" in s: return 1
    if "neutral" in s:  return 0
    return 0


def chatgpt_predict_with_confidence(tweet):
    """Plain fine-tuned gpt-4.1-mini, no retrieval. Confidence = exp(first-token logprob)."""
    response = client.chat.completions.create(
        model=FT_CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_BASE},
            {"role": "user",   "content": f"Classify the sentiment of this tweet: \"{tweet}\""},
        ],
        temperature=0.0, max_tokens=5,
        logprobs=True, top_logprobs=3,
    )
    pred = parse_label(response.choices[0].message.content)
    lp = response.choices[0].logprobs.content
    conf = math.exp(lp[0].logprob) if lp else 0.0
    return pred, conf


def dynamic_weighted_vote(preds_confs):
    """Rank by reported confidence; 0.50 to the top, 0.25 to each of the rest."""
    ranked = sorted(preds_confs, key=lambda x: x[2], reverse=True)
    weights = {}
    for i, (name, _, _) in enumerate(ranked):
        weights[name] = WEIGHT_TOP if i == 0 else (WEIGHT_MID if i == 1 else WEIGHT_BOT)

    scores = {-1: 0.0, 0: 0.0, 1: 0.0}
    for name, pred, conf in preds_confs:
        scores[pred] += weights[name] * conf

    final = max(scores.items(), key=lambda x: x[1])[0]
    return final, scores, weights


def classify_one(tweet):
    tw_pred, tw_conf = encoder_predict_one(TW_TOK, TW_MODEL, tweet)
    rl_pred, rl_conf = encoder_predict_one(RL_TOK, RL_MODEL, tweet)
    gp_pred, gp_conf = chatgpt_predict_with_confidence(tweet)

    preds = [
        ("twitter_roberta", tw_pred, tw_conf),
        ("roberta_large",   rl_pred, rl_conf),
        ("chatgpt_ft",      gp_pred, gp_conf),
    ]
    final, scores, weights = dynamic_weighted_vote(preds)
    return {
        "tw_pred": tw_pred, "tw_conf": tw_conf,
        "rl_pred": rl_pred, "rl_conf": rl_conf,
        "gp_pred": gp_pred, "gp_conf": gp_conf,
        "final_pred": final,
        "score_neg": scores[-1],
        "score_neu": scores[0],
        "score_pos": scores[1],
        "weight_tw": weights["twitter_roberta"],
        "weight_rl": weights["roberta_large"],
        "weight_gp": weights["chatgpt_ft"],
        "top_model": max(preds, key=lambda x: x[2])[0],
    }


def main(limit=None):
    test_df = pd.read_csv(TEST_FILE)
    if limit:
        test_df = test_df.head(limit)
    tweets = test_df["tweet"].astype(str).tolist()
    y_true = test_df["sentiment"].astype(int).tolist()
    n = len(tweets)
    print(f"\nTest set: {n} tweets, MAX_WORKERS={MAX_WORKERS}")

    rows = [None] * n

    def work(i):
        try:
            r = classify_one(tweets[i])
        except Exception as e:
            print(f"[warn] tweet {i} failed: {e}")
            r = {"tw_pred": 0, "tw_conf": 0.0, "rl_pred": 0, "rl_conf": 0.0,
                 "gp_pred": 0, "gp_conf": 0.0, "final_pred": 0,
                 "score_neg": 0.0, "score_neu": 0.0, "score_pos": 0.0,
                 "weight_tw": 0.0, "weight_rl": 0.0, "weight_gp": 0.0,
                 "top_model": "none"}
        r.update({"tweet": tweets[i], "true_label": y_true[i]})
        return i, r

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(work, i): i for i in range(n)}
        for fut in tqdm(as_completed(futures), total=n, desc="dynamic (no RAG)"):
            i, row = fut.result()
            rows[i] = row

    out_df = pd.DataFrame(rows)
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(RESULTS_FILE, index=False)

    y_pred = out_df["final_pred"].astype(int).tolist()
    print("\n" + "=" * 64)
    print("DYNAMIC CONFIDENCE-RANKED VOTING (0.50 / 0.25 / 0.25) — NO RAG")
    print("Voters: Twitter-RoBERTa, RoBERTa-Large, gpt-4.1-mini (fine-tuned)")
    print("=" * 64)
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Macro F1: {f1_score(y_true, y_pred, average='macro'):.4f}")
    print(classification_report(y_true, y_pred,
                                labels=[-1, 0, 1],
                                target_names=["Negative", "Neutral", "Positive"],
                                zero_division=0))

    print("Component accuracy vs mean reported confidence:")
    for c, name in [("tw", "twitter_roberta"), ("rl", "roberta_large"), ("gp", "chatgpt_ft")]:
        acc = accuracy_score(y_true, out_df[f"{c}_pred"].astype(int))
        conf = out_df[f"{c}_conf"]
        sat = (conf >= 0.99).mean()
        print(f"  {name:<18} acc={acc:.4f}  mean_conf={conf.mean():.4f}  "
              f"median={conf.median():.4f}  saturation(>=0.99)={100*sat:.1f}%")

    print("\nWhich model is ranked most confident (receives w=0.50):")
    for m, c in out_df["top_model"].value_counts().items():
        print(f"  {m:<18} {c:>4} ({100 * c / n:.1f}%)")

    print(f"\nSaved: {RESULTS_FILE}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None, help="Run on first N tweets")
    args = p.parse_args()
    main(limit=args.limit)
