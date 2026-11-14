import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

DATA = Path(__file__).resolve().parents[1] / "data"
SRC = DATA / "tweets_labeled.csv"

df = pd.read_csv(SRC)
df = df.rename(columns={"Unnamed: 0": "tweet", "Final score": "sentiment"})
df = df[["tweet", "sentiment"]].dropna()
df["sentiment"] = df["sentiment"].astype(int)

print(f"Total: {len(df)} rows")
print(f"Distribution:\n{df['sentiment'].value_counts().sort_index()}")

train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["sentiment"]
)

train_df.to_csv(DATA / "train.csv", index=False)
test_df.to_csv(DATA / "test.csv", index=False)

print(f"\nTrain: {len(train_df)} | Test: {len(test_df)}")
print(f"Saved to {DATA / 'train.csv'} and {DATA / 'test.csv'}")
