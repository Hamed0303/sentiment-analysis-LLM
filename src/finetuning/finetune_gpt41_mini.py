# =============================================================================
# FINE-TUNE gpt-4.1-mini (OpenAI fine-tuning API) — 999-tweet dataset
# =============================================================================
# Runs locally. Reads data/train.csv and data/test.csv
# (run src/split_dataset.py first to regenerate them).
#
# Workflow:
#   1) First run: trains and prints fine-tuned model ID
#   2) Save the model ID to .env as: FINETUNED_MODEL_HF=ft:gpt-4.1-mini-...
#   3) Re-run: evaluates on test.csv and saves results CSV
# =============================================================================

import csv
import json
import os
import re
import time
import requests
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FINETUNE_MODEL = "gpt-4.1-mini-2025-04-14"
FINETUNED_MODEL = os.getenv("FINETUNED_MODEL_HF")
client = OpenAI(api_key=OPENAI_API_KEY)

ROOT = Path(__file__).resolve().parents[2]
TRAIN_FILE = ROOT / "data" / "train.csv"
TEST_FILE = ROOT / "data" / "test.csv"
TRAINING_JSONL = ROOT / "data" / "gpt_finetune_train.jsonl"
RESULTS_FILE = ROOT / "results" / "predictions" / "gpt41_mini.csv"


def parse_finetuned_label(raw: str) -> int:
    if not raw:
        return 0
    s = raw.strip().lower().replace("−", "-").splitlines()[0]
    if re.fullmatch(r"-1|0|1", s):
        return int(s)
    m = re.search(
        r"(?<![0-9])-1(?![0-9])|(?<![0-9.])0(?![0-9])|(?<![0-9.])1(?![0-9])", s
    )
    if m:
        return int(m.group(0))
    if re.search(r"\bnegative\b", s):
        return -1
    if re.search(r"\bpositive\b", s):
        return 1
    if re.search(r"\bneutral\b", s):
        return 0
    return 0


def create_training_data():
    data = []
    with open(TRAIN_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                label = int(row["sentiment"])
                if label in [-1, 0, 1]:
                    data.append({"text": row["tweet"], "label": label})
            except (ValueError, KeyError):
                continue
    with open(TRAINING_JSONL, "w", encoding="utf-8") as f:
        for item in data:
            example = {
                "messages": [
                    {"role": "system", "content": "You are a sentiment classifier. Classify tweets as -1 (negative), 0 (neutral), or 1 (positive). Reply with only the number."},
                    {"role": "user", "content": f"Classify the sentiment of this tweet: \"{item['text']}\""},
                    {"role": "assistant", "content": str(item["label"])}
                ]
            }
            f.write(json.dumps(example) + "\n")
    print(f"Created {len(data)} training examples -> {TRAINING_JSONL}")


def upload_training_file():
    with open(TRAINING_JSONL, "rb") as f:
        response = requests.post(
            "https://api.openai.com/v1/files",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            files={"file": f},
            data={"purpose": "fine-tune"}
        )
    if response.status_code != 200:
        print(f"Upload failed: {response.text}")
        return None
    file_id = response.json().get("id")
    print(f"Uploaded! File ID: {file_id}")
    return file_id


def create_finetune_job(file_id):
    response = requests.post(
        "https://api.openai.com/v1/fine_tuning/jobs",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "training_file": file_id,
            "model": FINETUNE_MODEL,
            "hyperparameters": {"n_epochs": 3}
        }
    )
    if response.status_code != 200:
        print(f"Job creation failed: {response.text}")
        return None
    job_id = response.json().get("id")
    print(f"Fine-tuning job created! Job ID: {job_id}")
    return job_id


def wait_for_completion(job_id):
    print("Waiting for fine-tuning to complete (~15-30 mins)...")
    url = f"https://api.openai.com/v1/fine_tuning/jobs/{job_id}"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    while True:
        response = requests.get(url, headers=headers)
        if not response.text.strip():
            time.sleep(60)
            continue
        job = response.json()
        status = job.get("status", "unknown")
        print(f"Status: {status}")
        if status == "succeeded":
            model_id = job["fine_tuned_model"]
            print(f"Done! Model ID: {model_id}")
            return model_id
        elif status in ("failed", "cancelled"):
            print(f"Failed: {job.get('error', 'unknown error')}")
            return None
        time.sleep(60)


def classify_one(idx_tweet_label):
    idx, tweet, true_label = idx_tweet_label
    response = client.chat.completions.create(
        model=FINETUNED_MODEL,
        messages=[
            {"role": "system", "content": "You are a sentiment classifier. Classify tweets as -1 (negative), 0 (neutral), or 1 (positive). Reply with only the number."},
            {"role": "user", "content": f"Classify the sentiment of this tweet: \"{tweet}\""}
        ],
        temperature=0.0,
        max_tokens=5
    )
    pred = parse_finetuned_label(response.choices[0].message.content)
    return idx, tweet, true_label, pred


def evaluate():
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    df = pd.read_csv(TEST_FILE)
    tasks = [(i, str(row["tweet"]), int(row["sentiment"])) for i, row in df.iterrows()]

    results = [None] * len(tasks)
    with ThreadPoolExecutor(max_workers=20) as executor:
        for idx, tweet, true_label, pred in tqdm(
            executor.map(classify_one, tasks), total=len(tasks), desc="Evaluating"
        ):
            results[idx] = {"tweet": tweet, "true_label": true_label, "pred_label": pred}

    out_df = pd.DataFrame(results)
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(RESULTS_FILE, index=False)

    y_true = out_df["true_label"].tolist()
    y_pred = out_df["pred_label"].tolist()
    print(f"\nAccuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Macro F1: {f1_score(y_true, y_pred, average='macro'):.4f}")
    print(classification_report(y_true, y_pred, target_names=["Negative", "Neutral", "Positive"]))
    print(f"Saved to: {RESULTS_FILE}")


def train():
    print("=== ChatGPT Fine-tuning — hugging_face dataset ===")
    create_training_data()
    file_id = upload_training_file()
    if not file_id:
        return
    job_id = create_finetune_job(file_id)
    if not job_id:
        return
    model_id = wait_for_completion(job_id)
    if model_id:
        print(f"\nSave this to your .env as:\nFINETUNED_MODEL_HF={model_id}")
        print("Then re-run this script to evaluate.")


if __name__ == "__main__":
    if FINETUNED_MODEL:
        print(f"Found FINETUNED_MODEL_HF={FINETUNED_MODEL}")
        print("Evaluating on test set...\n")
        evaluate()
    else:
        print("No FINETUNED_MODEL_HF in .env — starting fine-tuning job.\n")
        train()
