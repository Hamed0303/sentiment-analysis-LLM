"""RAG knowledge-base comparison — always-on retrieval, both KBs.

For each of the 200 test tweets:
  1. Embed the tweet with text-embedding-3-small
  2. Retrieve top-K chunks from KB A (sentiment_kb.pdf — original hand-authored)
     and KB B (knowledge_bases/kb_b/ — five corpus-derived PDFs)
  3. Classify with fine-tuned gpt-4.1-mini twice (once per KB), each time
     using the retrieved chunks as system-prompt guidance
  4. Report accuracy and macro F1 for each KB

No confidence gating — every tweet hits the KB so the comparison isolates
KB content rather than routing policy. This is the clean apples-to-apples
KB-vs-KB test.
"""

import math
import os
import pickle
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import faiss
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHAT_MODEL = os.getenv("FINETUNED_MODEL_HF")
EMBED_MODEL = "text-embedding-3-small"

TOP_K = 3
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
MAX_WORKERS = 20

if not CHAT_MODEL:
    raise RuntimeError("FINETUNED_MODEL_HF not set in .env")

client = OpenAI(api_key=OPENAI_API_KEY)

ROOT = Path(__file__).resolve().parents[2]
TEST_FILE = ROOT / "data" / "test.csv"
KB_A_PDF = ROOT / "knowledge_bases" / "kb_a" / "sentiment_kb.pdf"
KB_A_CACHE = ROOT / "knowledge_bases" / "kb_a" / "kb_a_openai_index_cache.pkl"
KBV2_FOLDER = ROOT / "knowledge_bases" / "kb_b"
KB_B_CACHE = KBV2_FOLDER / "kbv2_openai_index_cache.pkl"
RESULTS_DIR = ROOT / "results" / "predictions"

SYSTEM_PROMPT_BASE = (
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


def embed_batch(texts, batch_size=100):
    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        out.extend([d.embedding for d in resp.data])
    return np.array(out, dtype=np.float32)


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks, i = [], 0
    while i < len(text):
        chunks.append(text[i:i + size])
        i += size - overlap
    return chunks


def build_kb_a():
    if KB_A_CACHE.exists():
        with open(KB_A_CACHE, "rb") as f:
            return pickle.load(f)
    if not KB_A_PDF.exists():
        raise FileNotFoundError(f"{KB_A_PDF} not found. Run create_sentiment_kb.py first.")
    print(f"Building KB A from {KB_A_PDF.name}...")
    text = "\n".join((p.extract_text() or "") for p in PdfReader(str(KB_A_PDF)).pages)
    chunks = chunk_text(text)
    embeddings = embed_batch(chunks)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    payload = (chunks, index)
    with open(KB_A_CACHE, "wb") as f:
        pickle.dump(payload, f)
    print(f"  KB A: {len(chunks)} chunks cached\n")
    return payload


def build_kb_b():
    if KB_B_CACHE.exists():
        with open(KB_B_CACHE, "rb") as f:
            return pickle.load(f)
    pdf_paths = sorted(KBV2_FOLDER.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found in {KBV2_FOLDER}")
    print(f"Building KB B from {len(pdf_paths)} PDFs in {KBV2_FOLDER.name}/...")
    all_chunks = []
    for pdf_path in pdf_paths:
        text = "\n".join((p.extract_text() or "") for p in PdfReader(str(pdf_path)).pages)
        all_chunks.extend(chunk_text(text))
    embeddings = embed_batch(all_chunks)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    payload = (all_chunks, index)
    with open(KB_B_CACHE, "wb") as f:
        pickle.dump(payload, f)
    print(f"  KB B: {len(all_chunks)} chunks cached\n")
    return payload


def classify_with_rag(tweet, kb_chunks, kb_index):
    tweet_vec = embed_batch([tweet])
    faiss.normalize_L2(tweet_vec)
    _, idx = kb_index.search(tweet_vec, TOP_K)
    retrieved = "\n\n".join(f"- {kb_chunks[i].strip()}" for i in idx[0])
    augmented_system = (
        SYSTEM_PROMPT_BASE +
        "\n\nRelevant sentiment classification guidance:\n" +
        retrieved +
        "\n\nUse this guidance to classify carefully. Reply with only -1, 0, or 1."
    )
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": augmented_system},
            {"role": "user", "content": f"Classify the sentiment of this tweet: \"{tweet}\""},
        ],
        temperature=0.0,
        max_tokens=5,
    )
    return parse_label(response.choices[0].message.content)


def evaluate_kb(name, kb_chunks, kb_index, tweets, y_true, out_file):
    print(f"\nClassifying 200 tweets with {name}...")
    preds = [None] * len(tweets)

    def work(i):
        return i, classify_with_rag(tweets[i], kb_chunks, kb_index)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for i, pred in tqdm(ex.map(work, range(len(tweets))), total=len(tweets), desc=name):
            preds[i] = pred

    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds, average="macro")
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.4f}    Macro F1: {f1:.4f}")
    print(classification_report(y_true, preds, target_names=["Negative", "Neutral", "Positive"]))

    pd.DataFrame({
        "tweet": tweets, "true_label": y_true, "pred_label": preds
    }).to_csv(out_file, index=False)
    print(f"Saved: {out_file}")
    return acc, f1


def main():
    test_df = pd.read_csv(TEST_FILE)
    tweets = test_df["tweet"].astype(str).tolist()
    y_true = test_df["sentiment"].astype(int).tolist()
    print(f"Test set: {len(tweets)} tweets")

    kb_a_chunks, kb_a_index = build_kb_a()
    kb_b_chunks, kb_b_index = build_kb_b()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    acc_a, f1_a = evaluate_kb(
        "KB A (sentiment_kb.pdf)", kb_a_chunks, kb_a_index, tweets, y_true,
        RESULTS_DIR / "rag_kb_a.csv",
    )
    acc_b, f1_b = evaluate_kb(
        "KB B (corpus-derived)", kb_b_chunks, kb_b_index, tweets, y_true,
        RESULTS_DIR / "rag_kb_b.csv",
    )

    print("\n" + "=" * 60)
    print("KB-vs-KB SUMMARY (always-on RAG, 200 test tweets)")
    print("=" * 60)
    print(f"{'':<32}{'Accuracy':>12}{'Macro F1':>12}")
    print(f"{'KB A (sentiment_kb.pdf)':<32}{acc_a:>12.4f}{f1_a:>12.4f}")
    print(f"{'KB B (corpus-derived)':<32}{acc_b:>12.4f}{f1_b:>12.4f}")
    print(f"{'delta (B - A)':<32}{acc_b - acc_a:>+12.4f}{f1_b - f1_a:>+12.4f}")


if __name__ == "__main__":
    main()
