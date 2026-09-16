
from pathlib import Path
import pandas as pd
import kagglehub

BRAND = "AppleSupport"
OUT = Path("data")
OUT.mkdir(parents=True, exist_ok=True)

# Download/load dataset
dataset_path = kagglehub.dataset_download(
    "thoughtvector/customer-support-on-twitter"
)

csvs = list(Path(dataset_path).rglob("*.csv"))
csv = max(csvs, key=lambda p: p.stat().st_size)

print("Using:", csv)

df = pd.read_csv(csv)

# Find AppleSupport tweets
brand_tweets = df[
    df["author_id"].astype(str).str.lower() == BRAND.lower()
].copy()

print("AppleSupport tweets:", len(brand_tweets))

# IMPORTANT:
# response_tweet_id contains the tweet IDs that this tweet responds to.
# We want customer tweets whose response is an AppleSupport tweet.
brand_ids = set(brand_tweets["tweet_id"].astype(str))

customer = df[
    df["response_tweet_id"].astype(str).isin(brand_ids)
].copy()

customer["text"] = customer["text"].astype(str).str.strip()
customer = customer[customer["text"].str.len() > 5]
customer = customer.drop_duplicates("text")

print("Customer messages found:", len(customer))

if len(customer) == 0:
    raise RuntimeError(
        "No customer messages found. Check the dataset relationship columns."
    )

# Development sample
sample = customer.sample(
    n=min(800, len(customer)),
    random_state=42
)

sample[["tweet_id", "created_at", "text"]].to_csv(
    OUT / "sample_data.csv",
    index=False
)

# Create golden set only if it doesn't already contain 200 rows
golden_path = OUT / "golden_200.csv"

if golden_path.exists():
    existing = pd.read_csv(golden_path)

    if len(existing) >= 150:
        print("Existing golden set found:", len(existing), "rows")
    else:
        golden = sample.sample(
            n=min(200, len(sample)),
            random_state=123
        )[["tweet_id", "text"]]

        golden["intent"] = ""
        golden["should_escalate"] = ""
        golden["annotator_a_intent"] = ""
        golden["annotator_a_escalate"] = ""
        golden["annotator_b_intent"] = ""
        golden["annotator_b_escalate"] = ""

        golden.to_csv(golden_path, index=False)
        print("Created new golden set:", len(golden))
else:
    golden = sample.sample(
        n=min(200, len(sample)),
        random_state=123
    )[["tweet_id", "text"]]

    golden["intent"] = ""
    golden["should_escalate"] = ""
    golden["annotator_a_intent"] = ""
    golden["annotator_a_escalate"] = ""
    golden["annotator_b_intent"] = ""
    golden["annotator_b_escalate"] = ""

    golden.to_csv(golden_path, index=False)
    print("Created golden set:", len(golden))

print("Saved", len(sample), "development rows.")
print("Golden file:", golden_path)
