"""Static confidence-weighted voting, top-heavy weights, WITHOUT RAG.

Identical to static_weighted_60_pipeline.py except that the third voter is
plain fine-tuned gpt-4.1-mini rather than the RAG-augmented variant. The voter
set is therefore the same as the static-tier (0.50/0.25/0.25) scheme, so the
two differ only in their weight vectors.

  - Twitter-RoBERTa (strongest single model)  -> weight 0.60
  - RoBERTa-Large                             -> weight 0.20
  - gpt-4.1-mini (fine-tuned, no retrieval)   -> weight 0.20

    score(c) = sum_m ( weight_m * conf_m * 1[pred_m = c] )

Final label = argmax_c score(c).

Output: results/predictions/static_top_heavy_voting.csv
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

WEIGHT_TWITTER = 0.60
WEIGHT_ROBERTA_LARGE = 0.20
WEIGHT_CHATGPT = 0.20

if not FT_CHAT_MODEL:
    raise RuntimeError("FINETUNED_MODEL_HF not set in .env")
if HF_TOKEN:
    login(token=HF_TOKEN)

client = OpenAI(api_key=OPENAI_API_KEY)
device = "cuda" if torch.cuda.is_available() else "cpu"

ROOT = Path(__file__).resolve().parents[2]
TEST_FILE = ROOT / "data" / "test.csv"
RESULTS_FILE = ROOT / "results" / "predictions" / "static_top_heavy_voting.csv"

REVERSE_MAP = {0: -1, 1: 0, 2: 1}

print(f"[init] Device: {device}")
print(f"[init] Fine-tuned classifier: {FT_CHAT_MODEL}")
print(f"[init] Weights: Twitter-RoBERTa={WEIGHT_TWITTER}, "
      f"RoBERTa-Large={WEIGHT_ROBERTA_LARGE}, ChatGPT(no RAG)={WEIGHT_CHATGPT}")

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


def static_weighted_vote(tw_pred, tw_conf, rl_pred, rl_conf, gp_pred, gp_conf):
    """Fixed weights per model. score[c] = sum(weight * conf * 1[pred = c])."""
    scores = {-1: 0.0, 0: 0.0, 1: 0.0}
    scores[tw_pred] += WEIGHT_TWITTER * tw_conf
    scores[rl_pred] += WEIGHT_ROBERTA_LARGE * rl_conf
    scores[gp_pred] += WEIGHT_CHATGPT * gp_conf
    final = max(scores.items(), key=lambda x: x[1])[0]
    return final, scores


def classify_one(tweet):
    tw_pred, tw_conf = encoder_predict_one(TW_TOK, TW_MODEL, tweet)
    rl_pred, rl_conf = encoder_predict_one(RL_TOK, RL_MODEL, tweet)
    gp_pred, gp_conf = chatgpt_predict_with_confidence(tweet)
    final, scores = static_weighted_vote(tw_pred, tw_conf, rl_pred, rl_conf, gp_pred, gp_conf)
    return {
        "tw_pred": tw_pred, "tw_conf": tw_conf,
        "rl_pred": rl_pred, "rl_conf": rl_conf,
        "gp_pred": gp_pred, "gp_conf": gp_conf,
        "final_pred": final,
        "score_neg": scores[-1],
        "score_neu": scores[0],
        "score_pos": scores[1],
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
                 "score_neg": 0.0, "score_neu": 0.0, "score_pos": 0.0}
        r.update({"tweet": tweets[i], "true_label": y_true[i]})
        return i, r

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(work, i): i for i in range(n)}
        for fut in tqdm(as_completed(futures), total=n, desc="static-60 (no RAG)"):
            i, row = fut.result()
            rows[i] = row

    out_df = pd.DataFrame(rows)
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(RESULTS_FILE, index=False)

    y_pred = out_df["final_pred"].astype(int).tolist()
    print("\n" + "=" * 62)
    print("STATIC WEIGHTED VOTING (0.60 / 0.20 / 0.20) — NO RAG")
    print("Voters: Twitter-RoBERTa, RoBERTa-Large, gpt-4.1-mini (fine-tuned)")
    print("=" * 62)
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Macro F1: {f1_score(y_true, y_pred, average='macro'):.4f}")
    print(classification_report(y_true, y_pred,
                                labels=[-1, 0, 1],
                                target_names=["Negative", "Neutral", "Positive"],
                                zero_division=0))

    print("Per-model standalone accuracy:")
    for col, name in [("tw_pred", f"twitter_roberta (w={WEIGHT_TWITTER})"),
                      ("rl_pred", f"roberta_large   (w={WEIGHT_ROBERTA_LARGE})"),
                      ("gp_pred", f"chatgpt_ft      (w={WEIGHT_CHATGPT})")]:
        acc = accuracy_score(y_true, out_df[col].astype(int))
        print(f"  {name:<34} {acc:.4f}")

    tw = out_df["tw_pred"].astype(int)
    fin = out_df["final_pred"].astype(int)
    yt = pd.Series(y_true)
    ov = tw != fin
    print(f"\nOverrides vs Twitter-RoBERTa: {int(ov.sum())} of {n}")
    print(f"  ensemble correct on overrides: {int((fin[ov] == yt[ov]).sum())}")
    print(f"  Twitter-RoBERTa correct on overrides: {int((tw[ov] == yt[ov]).sum())}")
    print(f"\nSaved: {RESULTS_FILE}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None, help="Run on first N tweets")
    args = p.parse_args()
    main(limit=args.limit)
