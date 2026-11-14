# =============================================================================
# FINE-TUNE ROBERTA-LARGE — hugging_face dataset (999 tweets)
# =============================================================================
# Runtime: L4 GPU | Training time: ~5-8 minutes
# Encoder model — fine-tunes classification head directly, no LoRA
# Upload train.csv and test.csv from data/ (produced by src/split_dataset.py)
# =============================================================================

# !pip install -q transformers datasets accelerate scikit-learn huggingface_hub

import torch
import numpy as np
import pandas as pd
from huggingface_hub import login, HfApi
from google.colab import files
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score, f1_score
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)

import os
HF_TOKEN = os.environ.get("HF_TOKEN")  # in Colab: add HF_TOKEN under Secrets, or set os.environ["HF_TOKEN"]
if HF_TOKEN is None:
    from google.colab import userdata
    HF_TOKEN = userdata.get("HF_TOKEN")
BASE_MODEL = "roberta-large"
HUB_MODEL_NAME = "Hamed0303/roberta-large-hf-finetuned"

login(token=HF_TOKEN)

print("Upload train.csv and test.csv")
uploaded = files.upload()

train_df_full = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Carve a 10% validation set out of train.csv (stratified, seed=42).
# Val is used for early stopping / best-checkpoint selection so the test
# set remains a true held-out partition (no leakage).
train_df, val_df = train_test_split(
    train_df_full, test_size=0.10, random_state=42,
    stratify=train_df_full["sentiment"]
)
print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
print(f"Train distribution:\n{train_df['sentiment'].value_counts().sort_index()}")
print(f"Val distribution:\n{val_df['sentiment'].value_counts().sort_index()}")

label_map = {-1: 0, 0: 1, 1: 2}
reverse_map = {0: -1, 1: 0, 2: 1}
id2label = {0: "negative", 1: "neutral", 2: "positive"}
label2id = {"negative": 0, "neutral": 1, "positive": 2}

train_df = train_df.assign(label=train_df["sentiment"].map(label_map))
val_df = val_df.assign(label=val_df["sentiment"].map(label_map))
test_df["label"] = test_df["sentiment"].map(label_map)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

def tokenize(batch):
    return tokenizer(batch["tweet"], truncation=True, max_length=128, padding="max_length")

train_dataset = Dataset.from_pandas(train_df[["tweet", "label"]])
val_dataset   = Dataset.from_pandas(val_df[["tweet", "label"]])
test_dataset  = Dataset.from_pandas(test_df[["tweet", "label"]])
train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset   = val_dataset.map(tokenize, batched=True)
test_dataset  = test_dataset.map(tokenize, batched=True)
train_dataset = train_dataset.rename_column("label", "labels")
val_dataset   = val_dataset.rename_column("label", "labels")
test_dataset  = test_dataset.rename_column("label", "labels")
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format("torch",   columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format("torch",  columns=["input_ids", "attention_mask", "labels"])

model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL, num_labels=3, id2label=id2label, label2id=label2id
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }

class_weights = torch.tensor(
    compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=train_df["label"].values),
    dtype=torch.float32,
)

class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        w = self.class_weights.to(logits.device)
        loss = torch.nn.CrossEntropyLoss(weight=w)(
            logits.view(-1, model.config.num_labels), labels.view(-1)
        )
        return (loss, outputs) if return_outputs else loss

training_args = TrainingArguments(
    output_dir="./roberta_large_hf_finetuned",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=50,
    lr_scheduler_type="cosine",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    fp16=True,
    logging_steps=20,
    save_total_limit=1,
    report_to="none",
)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
    class_weights=class_weights.tolist(),
)

print("Starting fine-tuning...")
trainer.train()
print("Training complete!")

print(f"Pushing to HuggingFace Hub: {HUB_MODEL_NAME}")
HfApi().create_repo(HUB_MODEL_NAME, token=HF_TOKEN, exist_ok=True)
model.push_to_hub(HUB_MODEL_NAME, token=HF_TOKEN)
tokenizer.push_to_hub(HUB_MODEL_NAME, token=HF_TOKEN)

print("Evaluating on test set...")
preds_output = trainer.predict(test_dataset)
raw_preds = np.argmax(preds_output.predictions, axis=-1)

y_true = test_df["sentiment"].tolist()
y_pred = [reverse_map[p] for p in raw_preds]

print(f"\nAccuracy: {accuracy_score(y_true, y_pred):.4f}")
print(f"Macro F1: {f1_score(y_true, y_pred, average='macro'):.4f}")
print(classification_report(y_true, y_pred, target_names=["Negative", "Neutral", "Positive"]))

results_df = test_df.copy()
results_df["pred_label"] = y_pred
results_df.to_csv("roberta_large.csv", index=False)
files.download("roberta_large.csv")
print("Done!")
