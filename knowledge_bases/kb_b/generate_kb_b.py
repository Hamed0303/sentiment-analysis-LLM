import pandas as pd
import re
from collections import Counter
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_LEFT

# KB B is derived from a separate, related corpus of 3,895 labelled social-media
# captions (columns: tweet, sentiment), NOT from the 999-tweet study dataset.
# That corpus is not redistributed here; the generated PDFs in this folder are.
OUT_DIR = Path(__file__).parent
TRAIN_FILE = OUT_DIR / "related_corpus_train.csv"

STOP_WORDS = {
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it", "they",
    "them", "this", "that", "a", "an", "the", "and", "or", "but", "in", "on",
    "at", "to", "for", "of", "with", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "did", "will", "would", "can", "could", "not",
    "no", "so", "if", "as", "by", "from", "up", "out", "just", "about", "when",
    "what", "who", "how", "all", "more", "than", "then", "now", "get", "got",
    "its", "it's", "via", "rt", "amp", "http", "https", "www", "co", "t", "s"
}


def extract_hashtags(text):
    return re.findall(r'#\w+', str(text).lower())


def extract_words(text):
    words = re.findall(r'\b[a-zA-Z]{3,}\b', str(text).lower())
    return [w for w in words if w not in STOP_WORDS]


def build_doc(filename, title, sections):
    doc = SimpleDocTemplate(
        str(OUT_DIR / filename),
        pagesize=letter,
        rightMargin=inch, leftMargin=inch,
        topMargin=inch, bottomMargin=inch
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("title", parent=styles["Heading1"], fontSize=16, spaceAfter=12)
    h2_style = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=13, spaceAfter=8)
    body_style = ParagraphStyle("body", parent=styles["Normal"], fontSize=10, spaceAfter=6, leading=14)

    story = [Paragraph(title, title_style), Spacer(1, 0.2 * inch)]
    for heading, paragraphs in sections:
        story.append(Paragraph(heading, h2_style))
        for para in paragraphs:
            safe = para.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            story.append(Paragraph(safe, body_style))
        story.append(Spacer(1, 0.15 * inch))

    doc.build(story)
    print(f"Generated: {filename}")


def generate_sentiment_pdf(df, label, label_name, filename):
    subset = df[df["sentiment"] == label].copy()
    all_words = []
    all_hashtags = []
    for text in subset["tweet"]:
        all_words.extend(extract_words(text))
        all_hashtags.extend(extract_hashtags(text))

    top_words = [w for w, _ in Counter(all_words).most_common(40)]
    top_hashtags = [h for h, _ in Counter(all_hashtags).most_common(30)]
    samples = subset["tweet"].dropna().sample(min(60, len(subset)), random_state=42).tolist()

    sections = [
        ("Overview", [
            f"This document contains {label_name} sentiment patterns extracted from a labeled social media dataset.",
            f"Total {label_name} examples in training set: {len(subset)}.",
            f"These patterns help identify {label_name} sentiment in tweets and social media captions."
        ]),
        ("Most Common Words in " + label_name + " Tweets", [
            "The following words frequently appear in " + label_name + " tweets:",
            ", ".join(top_words) + ".",
            "These words are strong indicators of " + label_name + " sentiment when they appear in text."
        ]),
        ("Common Hashtags Associated with " + label_name + " Sentiment", [
            "Hashtags are important signals in social media sentiment. The following hashtags commonly appear in " + label_name + " posts:",
            ", ".join(top_hashtags) if top_hashtags else "No dominant hashtags found.",
            "When these hashtags appear in a tweet, they are likely indicators of " + label_name + " sentiment."
        ]),
        ("Representative Examples of " + label_name + " Tweets", [
            "The following are real examples of " + label_name + " sentiment tweets from the dataset:"
        ] + [f"- {t.strip()}" for t in samples[:30]]),
        ("More " + label_name + " Examples", [
            f"- {t.strip()}" for t in samples[30:]
        ]),
        ("Classification Rules for " + label_name + " Sentiment", get_rules(label_name))
    ]

    build_doc(filename, f"{label_name} Sentiment Patterns — Social Media Dataset", sections)


def get_rules(label_name):
    rules = {
        "Positive": [
            "A tweet is POSITIVE (label=1) when it expresses joy, excitement, love, gratitude, pride, or satisfaction.",
            "Positive indicators: words like love, great, happy, excited, amazing, wonderful, good, beautiful, best, congrats.",
            "Positive hashtags: #excited, #happy, #love, #blessed, #grateful, #amazing, #proud, #winning.",
            "Exclamation marks and multiple positive adjectives together strongly indicate positive sentiment.",
            "Achievements, celebrations, compliments, and expressions of affection are positive.",
            "Even if describing something challenging (e.g., workout pain), if the overall tone is proud or motivated, classify as positive."
        ],
        "Negative": [
            "A tweet is NEGATIVE (label=-1) when it expresses anger, sadness, frustration, disappointment, disgust, or fear.",
            "Negative indicators: words like hate, terrible, awful, disgusting, failed, sick, broken, wrong, bad, worst.",
            "Negative hashtags: #disgusting, #disgusted, #shocked, #disgraced, #rejected, #aching, #remorse.",
            "Complaints about services, weather, health issues, or relationships are typically negative.",
            "Words like 'can't believe', 'so annoying', 'worst ever' are strong negative signals.",
            "Even mild negative expressions like '#aching' after a workout or '#remorse' after eating count as negative.",
            "Sarcasm with exaggerated positives (e.g., 'Best day ever' when things go wrong) should be treated as negative."
        ],
        "Neutral": [
            "A tweet is NEUTRAL (label=0) when it states facts, shares news, provides information, or lacks clear emotional tone.",
            "Neutral tweets describe events without emotional language: announcements, news articles, factual statements.",
            "Retweets (RT) of news or informational content are often neutral.",
            "Neutral tweets may contain hashtags that are topic markers rather than emotional signals.",
            "If a tweet has both positive and negative elements with no dominant tone, classify as neutral.",
            "Questions, generic observations, and promotional content without clear emotion tend to be neutral."
        ]
    }
    return rules.get(label_name, [])


def generate_dataset_overview(df):
    counts = df["sentiment"].value_counts()
    label_map = {-1: "Negative", 0: "Neutral", 1: "Positive"}

    avg_lengths = df.groupby("sentiment")["tweet"].apply(lambda x: x.str.len().mean())

    sections = [
        ("Dataset Statistics", [
            f"Total training examples: {len(df)}",
            f"Positive tweets (label=1): {counts.get(1, 0)} ({counts.get(1, 0)/len(df)*100:.1f}%)",
            f"Negative tweets (label=-1): {counts.get(-1, 0)} ({counts.get(-1, 0)/len(df)*100:.1f}%)",
            f"Neutral tweets (label=0): {counts.get(0, 0)} ({counts.get(0, 0)/len(df)*100:.1f}%)",
        ]),
        ("Average Tweet Length by Sentiment", [
            f"Positive: {avg_lengths.get(1, 0):.0f} characters",
            f"Negative: {avg_lengths.get(-1, 0):.0f} characters",
            f"Neutral: {avg_lengths.get(0, 0):.0f} characters",
        ]),
        ("Domain Description", [
            "This dataset consists of social media captions and tweets covering a wide range of everyday topics.",
            "Topics include: fitness and gym, relationships and love, food, travel, news events, sports, technology, and personal life.",
            "Many tweets use hashtags as topic and emotion markers. Hashtags are critical signals for sentiment classification.",
            "The text often contains informal language, abbreviations, slang, emojis, and @mentions.",
            "Label mapping used: -1 = Negative, 0 = Neutral, 1 = Positive.",
        ]),
        ("Sentiment Classification Guidelines", [
            "Use the full context of the tweet — hashtags, emojis, and surrounding words all contribute to sentiment.",
            "Do not rely on a single word; consider the overall tone and intent.",
            "Sarcasm and irony often reverse the apparent sentiment — look for contextual clues like 'Best day ever' after negative events.",
            "Negation changes sentiment: 'not bad' is positive, 'not good' is negative.",
            "Emoji can shift sentiment: heart emojis suggest positive, skull/crying emojis can suggest negative or humor.",
            "When in doubt between negative and neutral, consider whether there is any clear emotional language present."
        ])
    ]

    build_doc("Dataset_Overview.pdf", "Social Media Sentiment Dataset — Overview and Guidelines", sections)


def generate_hashtag_pdf(df):
    pos = df[df["sentiment"] == 1]
    neg = df[df["sentiment"] == -1]
    neu = df[df["sentiment"] == 0]

    def top_tags(subset, n=40):
        tags = []
        for text in subset["tweet"]:
            tags.extend(extract_hashtags(text))
        return Counter(tags).most_common(n)

    pos_tags = top_tags(pos)
    neg_tags = top_tags(neg)
    neu_tags = top_tags(neu)

    sections = [
        ("Positive Sentiment Hashtags", [
            "These hashtags most frequently appear in POSITIVE tweets (label=1):",
            ", ".join(f"{t}({c})" for t, c in pos_tags),
            "When any of these hashtags appear in a tweet, they are a strong signal of positive sentiment."
        ]),
        ("Negative Sentiment Hashtags", [
            "These hashtags most frequently appear in NEGATIVE tweets (label=-1):",
            ", ".join(f"{t}({c})" for t, c in neg_tags),
            "When any of these hashtags appear in a tweet, they are a strong signal of negative sentiment."
        ]),
        ("Neutral Sentiment Hashtags", [
            "These hashtags most frequently appear in NEUTRAL tweets (label=0):",
            ", ".join(f"{t}({c})" for t, c in neu_tags),
            "These hashtags are typically topic markers without strong emotional content."
        ]),
        ("Hashtag Classification Rules", [
            "If a tweet contains predominantly positive hashtags, classify as positive.",
            "If a tweet contains predominantly negative hashtags (e.g., #disgusting, #shocked, #rejected), classify as negative.",
            "Hashtags that describe emotions directly (e.g., #happy, #sad, #angry) are the strongest sentiment signals.",
            "Hashtags that describe topics without emotion (e.g., #news, #photo, #video) are neutral indicators.",
            "Multiple hashtags of the same sentiment reinforce each other — treat them as a combined signal."
        ])
    ]

    build_doc("Hashtag_Sentiment_Rules.pdf", "Hashtag Sentiment Patterns — Social Media Dataset", sections)


def main():
    df = pd.read_csv(TRAIN_FILE)
    df = df.dropna(subset=["tweet", "sentiment"])
    df["sentiment"] = df["sentiment"].astype(int)

    print(f"Loaded {len(df)} training examples")
    print(df["sentiment"].value_counts())

    generate_dataset_overview(df)
    generate_sentiment_pdf(df, 1, "Positive", "Positive_Sentiment_Patterns.pdf")
    generate_sentiment_pdf(df, -1, "Negative", "Negative_Sentiment_Patterns.pdf")
    generate_sentiment_pdf(df, 0, "Neutral", "Neutral_Sentiment_Patterns.pdf")
    generate_hashtag_pdf(df)

    print("\nAll PDFs generated in knowledge_bases/kb_b/")


if __name__ == "__main__":
    main()
