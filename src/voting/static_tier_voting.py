"""Weighted Voting Pipeline.


    score[class] = Σ (confidence_i × weight_i)   over models i that voted for that class
    final_prediction = argmax(score)


Default weights:
  Twitter-RoBERTa = 0.50  (the primary neural)
  RoBERTa-Large   = 0.25  (verifier 1)
  ChatGPT         = 0.25  (verifier 2)
.
"""

import math
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from openai import OpenAI
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHAT_MODEL = os.getenv("FINETUNED_MODEL_HF")
HF_TOKEN = os.getenv("Hugging_Face") or os.getenv("HF_TOKEN")

TWITTER_MODEL = "Hamed0303/twitter-roberta-hf-finetuned"
ROBERTA_MODEL = "Hamed0303/roberta-large-hf-finetuned"

# === WEIGHTS — must sum to 1.0 ===
NEURAL_WEIGHTS = {
    "twitter_roberta": 0.50,
    "roberta_large":   0.25,
    "chatgpt":         0.25,
}
assert abs(sum(NEURAL_WEIGHTS.values()) - 1.0) < 1e-6

MAX_WORKERS = 20

if not CHAT_MODEL:
    raise RuntimeError("FINETUNED_MODEL_HF not set in .env")
if HF_TOKEN:
    login(token=HF_TOKEN)

client = OpenAI(api_key=OPENAI_API_KEY)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}\n")

ROOT = Path(__file__).resolve().parents[2]
TEST_FILE = ROOT / "data" / "test.csv"
RESULTS_FILE = ROOT / "results" / "predictions" / "static_tier_voting.csv"

REVERSE_MAP = {0: -1, 1: 0, 2: 1}
CLASS_NAMES = {-1: "negative", 0: "neutral", 1: "positive"}

SYSTEM_PROMPT = (
    "You are a sentiment classifier. Classify tweets as -1 (negative), "
    "0 (neutral), or 1 (positive). Reply with only the number."
)


def parse_label(raw):
    if not raw:
        return 0
    s = raw.strip().lower().replace("−", "-").splitlines()[0]
    if re.fullmatch(r"-1|0|1", s):
        return int(s)
    m = re.search(r"(?<![0-9])-1(?![0-9])|(?<![0-9.])0(?![0-9])|(?<![0-9.])1(?![0-9])", s)
    if m:
        return int(m.group(0))
    if "negative" in s: return -1
    if "positive" in s: return 1
    if "neutral" in s: return 0
    return 0


def load_encoder(model_id):
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, token=HF_TOKEN).to(device)
    model.eval()
    return tokenizer, model


def encoder_predict_all(tokenizer, model, tweets, batch_size=16, desc=""):
    """Returns list of (pred_label, confidence) for every tweet."""
    out = []
    for i in tqdm(range(0, len(tweets), batch_size), desc=desc):
        batch = tweets[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True,
                           max_length=128, padding=True).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        for p in probs:
            out.append((REVERSE_MAP[p.argmax().item()], p.max().item()))
    return out


def chatgpt_predict(tweet):
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Classify the sentiment of this tweet: \"{tweet}\""},
        ],
        temperature=0.0,
        max_tokens=5,
        logprobs=True,
        top_logprobs=3,
    )
    pred = parse_label(response.choices[0].message.content)
    lp = response.choices[0].logprobs.content
    conf = math.exp(lp[0].logprob) if lp else 0.0
    return pred, conf


def weighted_vote(neural_preds):
    """
    neural_preds: {model_name: (pred, conf)}
    Returns: final_pred, scores dict, contributions list
    """
    scores = {-1: 0.0, 0: 0.0, 1: 0.0}
    contributions = []
    for name, (pred, conf) in neural_preds.items():
        weight = NEURAL_WEIGHTS[name]
        contribution = conf * weight
        scores[pred] += contribution
        contributions.append({
            "name": name, "pred": pred, "conf": round(conf, 3),
            "weight": weight, "contribution": round(contribution, 4),
        })
    final = max(scores, key=scores.get)
    return final, scores, contributions


def main():
    test_df = pd.read_csv(TEST_FILE)
    tweets = test_df["tweet"].astype(str).tolist()
    y_true = test_df["sentiment"].astype(int).tolist()
    n = len(tweets)
    print(f"Test set: {n} tweets\n")

    # ---- Twitter-RoBERTa ----
    tw_tok, tw_model = load_encoder(TWITTER_MODEL)
    tw_results = encoder_predict_all(tw_tok, tw_model, tweets, desc="Twitter-RoBERTa")
    del tw_model, tw_tok
    if device == "cuda":
        torch.cuda.empty_cache()

    # ---- RoBERTa-Large ----
    rl_tok, rl_model = load_encoder(ROBERTA_MODEL)
    rl_results = encoder_predict_all(rl_tok, rl_model, tweets, desc="RoBERTa-Large")
    del rl_model, rl_tok
    if device == "cuda":
        torch.cuda.empty_cache()

    # ---- ChatGPT ----
    print("ChatGPT predictions (parallel)...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        gpt_results = list(tqdm(ex.map(chatgpt_predict, tweets), total=n, desc="ChatGPT"))

    # ---- Weighted voting ----
    print("\nApplying weighted voting...")
    rows = []
    all_contributions = []
    for i, tweet in enumerate(tweets):
        neural_preds = {
            "twitter_roberta": tw_results[i],
            "roberta_large":   rl_results[i],
            "chatgpt":         gpt_results[i],
        }
        final, scores, contribs = weighted_vote(neural_preds)
        all_contributions.append(contribs)
        rows.append({
            "tweet": tweet,
            "true_label": y_true[i],
            "tw_pred": tw_results[i][0], "tw_conf": round(tw_results[i][1], 3),
            "rl_pred": rl_results[i][0], "rl_conf": round(rl_results[i][1], 3),
            "gpt_pred": gpt_results[i][0], "gpt_conf": round(gpt_results[i][1], 3),
            "score_neg": round(scores[-1], 4),
            "score_neu": round(scores[0], 4),
            "score_pos": round(scores[1], 4),
            "final_pred": final,
            "final_confidence": round(max(scores.values()), 4),
        })

    out_df = pd.DataFrame(rows)
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(RESULTS_FILE, index=False)

    y_pred = out_df["final_pred"].tolist()

    # ---- Report ----
    print("\n" + "=" * 60)
    print("WEIGHTED VOTING — Neural Ensemble")
    print("=" * 60)
    print("\nWeights:")
    for k, v in NEURAL_WEIGHTS.items():
        print(f"  {k:<20} {v:.2f}")

    print(f"\nFinal Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Final Macro F1: {f1_score(y_true, y_pred, average='macro'):.4f}")
    print(classification_report(y_true, y_pred, target_names=["Negative", "Neutral", "Positive"]))

    print("Standalone model accuracies:")
    for col, label in [("tw_pred", "Twitter-RoBERTa"),
                       ("rl_pred", "RoBERTa-Large"),
                       ("gpt_pred", "ChatGPT")]:
        acc = accuracy_score(y_true, out_df[col].tolist())
        print(f"  {label:<20} {acc:.4f}")

    # Agreement statistics
    unanimous = sum(
        1 for i in range(n)
        if tw_results[i][0] == rl_results[i][0] == gpt_results[i][0]
    )
    three_way_split = sum(
        1 for i in range(n)
        if len({tw_results[i][0], rl_results[i][0], gpt_results[i][0]}) == 3
    )
    print(f"\nAgreement: {unanimous}/{n} unanimous ({100*unanimous/n:.1f}%) | "
          f"{three_way_split}/{n} three-way split ({100*three_way_split/n:.1f}%)")

    # Interpretability examples (one per predicted class)
    print("\n" + "=" * 60)
    print("INTERPRETABILITY EXAMPLES (one per class)")
    print("=" * 60)
    for target in [-1, 0, 1]:
        example_idx = next((i for i, r in enumerate(rows) if r["final_pred"] == target), None)
        if example_idx is None:
            continue
        r = rows[example_idx]
        print(f"\nTweet: {r['tweet']!r}")
        print(f"True: {CLASS_NAMES[r['true_label']]} | "
              f"Predicted: {CLASS_NAMES[r['final_pred']]}")
        print(f"Scores: neg={r['score_neg']:.3f} | "
              f"neu={r['score_neu']:.3f} | pos={r['score_pos']:.3f}")
        print(f"{'Model':<18} {'Pred':<10} {'Conf':<8} {'Weight':<8} {'Contribution':<12}")
        for c in all_contributions[example_idx]:
            print(f"  {c['name']:<16} {CLASS_NAMES[c['pred']]:<10} "
                  f"{c['conf']:<8.3f} {c['weight']:<8.2f} {c['contribution']:<12.4f}")

    print(f"\nFull results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
