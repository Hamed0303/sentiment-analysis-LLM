"""LangGraph supervisor multi-agent pipeline WITHOUT the RAG tool (with hints).

Identical to supervisor_pipeline.py except that classify_with_rag has been
removed. The tool pool is now exactly the three voters used by the static
weighted-voting ensembles:

    - classify_with_twitter_roberta  (domain encoder)     weight 0.50
    - classify_with_roberta_large    (general encoder)    weight 0.25
    - classify_with_chatgpt          (fine-tuned LLM)     weight 0.25

Weights now sum to 1.00, and every tool reports a real confidence, so the
aggregator no longer needs a default-confidence rule.

Output: results/predictions/multi_agent_with_hints.csv

Usage:
  python multi_agent_langgraph/supervisor_pipeline_norag.py
  python multi_agent_langgraph/supervisor_pipeline_norag.py --limit 10
"""

import argparse
import json
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Annotated, Sequence, TypedDict
from operator import add

import pandas as pd
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from openai import OpenAI
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FT_CHAT_MODEL = os.getenv("FINETUNED_MODEL_HF")
SUPERVISOR_MODEL = "gpt-4.1-mini-2025-04-14"
HF_TOKEN = os.getenv("Hugging_Face") or os.getenv("HF_TOKEN")
TWITTER_MODEL = "Hamed0303/twitter-roberta-hf-finetuned"
ROBERTA_MODEL = "Hamed0303/roberta-large-hf-finetuned"

MAX_WORKERS = 4

if not FT_CHAT_MODEL:
    raise RuntimeError("FINETUNED_MODEL_HF not set in .env")
if HF_TOKEN:
    login(token=HF_TOKEN)

client = OpenAI(api_key=OPENAI_API_KEY)
device = "cuda" if torch.cuda.is_available() else "cpu"

ROOT = Path(__file__).resolve().parents[2]
TEST_FILE = ROOT / "data" / "test.csv"
RESULTS_FILE = ROOT / "results" / "predictions" / "multi_agent_with_hints.csv"

REVERSE_MAP = {0: -1, 1: 0, 2: 1}

print(f"[init] Device: {device}")
print(f"[init] Supervisor: {SUPERVISOR_MODEL}")
print(f"[init] Fine-tuned classifier: {FT_CHAT_MODEL}")
print("[init] Tools: twitter-roberta, roberta-large, chatgpt-ft  (no RAG)")

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
    response = client.chat.completions.create(
        model=FT_CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_BASE},
            {"role": "user", "content": f"Classify the sentiment of this tweet: \"{tweet}\""},
        ],
        temperature=0.0, max_tokens=5,
        logprobs=True, top_logprobs=3,
    )
    pred = parse_label(response.choices[0].message.content)
    lp = response.choices[0].logprobs.content
    conf = math.exp(lp[0].logprob) if lp else 0.0
    return pred, conf


@tool
def classify_with_twitter_roberta(tweet: str) -> str:
    """Classify a tweet using fine-tuned Twitter-RoBERTa, a domain-pretrained
    encoder optimised for tweet-level sentiment. Best for: short, slang-heavy,
    hashtag-laden tweets. Returns JSON with label (-1, 0, 1) and confidence."""
    pred, conf = encoder_predict_one(TW_TOK, TW_MODEL, tweet)
    return json.dumps({"model": "twitter-roberta", "label": pred,
                       "confidence": round(conf, 4)})


@tool
def classify_with_roberta_large(tweet: str) -> str:
    """Classify a tweet using fine-tuned RoBERTa-Large, a larger general-purpose
    encoder. Best for: longer, more formal-sounding tweets where general
    language understanding matters. Returns JSON with label (-1, 0, 1) and confidence."""
    pred, conf = encoder_predict_one(RL_TOK, RL_MODEL, tweet)
    return json.dumps({"model": "roberta-large", "label": pred,
                       "confidence": round(conf, 4)})


@tool
def classify_with_chatgpt(tweet: str) -> str:
    """Classify a tweet using fine-tuned gpt-4.1-mini. Best for: nuanced,
    sarcastic, or context-dependent tweets where deep language reasoning helps.
    Returns JSON with label (-1, 0, 1) and confidence."""
    pred, conf = chatgpt_predict_with_confidence(tweet)
    return json.dumps({"model": "chatgpt-ft", "label": pred,
                       "confidence": round(conf, 4)})


TOOLS = [classify_with_twitter_roberta, classify_with_roberta_large,
         classify_with_chatgpt]
TOOL_NODE = ToolNode(TOOLS)


class AgentState(TypedDict):
    tweet: str
    messages: Annotated[Sequence, add]
    final_label: int


SUPERVISOR_SYS = """You are a sentiment-classification supervisor. Given a tweet,
decide which classifier(s) to consult based on the tweet's characteristics.

Available tools (call one or more):
- classify_with_twitter_roberta - fast, best for short tweets, slang, hashtags
- classify_with_roberta_large - slower, better for longer/formal tweets
- classify_with_chatgpt - fine-tuned LLM, best for sarcasm/nuance

Strategy:
1. Read the tweet.
2. Call one or more tools that you judge most appropriate.
3. You MUST call at least one tool. Do not produce a final label yourself.
4. Be efficient: call only the tools you believe are genuinely needed.

After tools return, you will be asked separately to produce the final label."""


AGGREGATOR_SYS = """You are a deterministic sentiment classification aggregator. Your job is to calculate a final sentiment label (-1, 0, or 1) using a strict Weighted Confidence Score formula.

Model Weights:
- twitter-roberta: 0.50
- roberta-large: 0.25
- chatgpt-ft: 0.25

Instructions:
1. Identify which models responded, what label (-1, 0, 1) they predicted, and their confidence scores.
2. For each unique label predicted, calculate its total score using the formula:
   Score[label] = Sum of (Confidence_i * Weight_i) for all models 'i' that voted for that specific label.
3. The label with the highest final score wins.
4. Break flat numerical ties by favoring the label with the highest single confidence score, or by simple majority if identical.

You must do the math silently. Reply with ONLY the final winning number (-1, 0, or 1). Absolutely no text, no code blocks, and no explanation."""


def supervisor_node(state):
    llm = ChatOpenAI(model=SUPERVISOR_MODEL, temperature=0).bind_tools(TOOLS)
    msgs = [SystemMessage(content=SUPERVISOR_SYS)]
    if not state["messages"]:
        msgs.append(HumanMessage(content=f"Classify this tweet: \"{state['tweet']}\""))
    else:
        msgs.extend(state["messages"])
    response = llm.invoke(msgs)
    return {"messages": [response]}


def aggregator_node(state):
    llm = ChatOpenAI(model=SUPERVISOR_MODEL, temperature=0)
    tool_outputs = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    results = []
    for tm in tool_outputs:
        try:
            d = json.loads(tm.content)
            results.append(f"- {d['model']}: label={d['label']}, conf={d.get('confidence')}")
        except Exception:
            results.append(f"- raw: {tm.content[:100]}")
    summary = "\n".join(results) if results else "(no classifier results)"
    response = llm.invoke([
        SystemMessage(content=AGGREGATOR_SYS),
        HumanMessage(content=f"Tweet: \"{state['tweet']}\"\n\nClassifier predictions:\n{summary}"),
    ])
    return {"final_label": parse_label(response.content), "messages": [response]}


def should_continue(state):
    last = state["messages"][-1] if state["messages"] else None
    if last and getattr(last, "tool_calls", None):
        return "tools"
    return "aggregator"


workflow = StateGraph(AgentState)
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("tools", TOOL_NODE)
workflow.add_node("aggregator", aggregator_node)
workflow.set_entry_point("supervisor")
workflow.add_conditional_edges("supervisor", should_continue,
                               {"tools": "tools", "aggregator": "aggregator"})
workflow.add_edge("tools", "supervisor")
workflow.add_edge("aggregator", END)
GRAPH = workflow.compile()


def classify_one(tweet):
    state = {"tweet": tweet, "messages": [], "final_label": 0}
    try:
        result = GRAPH.invoke(state, config={"recursion_limit": 8})
    except Exception as e:
        print(f"[warn] graph failed on tweet {tweet[:60]!r}: {e}")
        return 0, [], "graph_error"
    tools_called = []
    for msg in result.get("messages", []):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tools_called.append(tc["name"])
    return result.get("final_label", 0), tools_called, "ok"


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
        pred, tools, status = classify_one(tweets[i])
        return i, {
            "tweet": tweets[i],
            "true_label": y_true[i],
            "pred_label": pred,
            "tools_called": ";".join(tools),
            "n_tools_called": len(tools),
            "status": status,
        }

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(work, i): i for i in range(n)}
        for fut in tqdm(as_completed(futures), total=n, desc="multi-agent (no RAG)"):
            i, row = fut.result()
            rows[i] = row

    out_df = pd.DataFrame(rows)
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(RESULTS_FILE, index=False)

    y_pred = out_df["pred_label"].tolist()
    print("\n" + "=" * 60)
    print("MULTI-AGENT SUPERVISOR — NO RAG TOOL (with hints)")
    print("Tools: twitter-roberta (0.50), roberta-large (0.25), chatgpt-ft (0.25)")
    print("=" * 60)
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Macro F1: {f1_score(y_true, y_pred, average='macro'):.4f}")
    print(classification_report(y_true, y_pred,
                                labels=[-1, 0, 1],
                                target_names=["Negative", "Neutral", "Positive"],
                                zero_division=0))

    print("Tool-usage statistics:")
    tool_counts = {}
    for tcalls in out_df["tools_called"]:
        for t in (tcalls.split(";") if tcalls else []):
            tool_counts[t] = tool_counts.get(t, 0) + 1
    for t, c in sorted(tool_counts.items(), key=lambda x: -x[1]):
        print(f"  {t:<40} {c:>4} ({100 * c / n:.1f}% of tweets)")
    print(f"\nAvg tools per tweet: {out_df['n_tools_called'].mean():.2f}")
    n_err = (out_df["status"] != "ok").sum()
    print(f"Graph errors: {n_err} / {n} ({100 * n_err / n:.1f}%)")
    print(f"Saved: {RESULTS_FILE}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None, help="Run on first N tweets")
    args = p.parse_args()
    main(limit=args.limit)
