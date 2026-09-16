
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics import cohen_kappa_score


def main():
    data_path = Path("data/golden_200.csv")
    df = pd.read_csv(data_path)

    # Keep rows with usable provisional intent labels
    df["intent"] = df["intent"].fillna("").astype(str).str.strip()
    df = df[df["intent"] != ""].copy()

    print("Evaluation rows:", len(df))

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"].astype(str),
        df["intent"],
        test_size=0.3,
        random_state=42,
        stratify=df["intent"]
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        max_features=5000,
        ngram_range=(1, 2)
    )

    Xtr = vectorizer.fit_transform(X_train)
    Xte = vectorizer.transform(X_test)

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    model.fit(Xtr, y_train)

    predictions = model.predict(Xte)

    results = {
        "rows": int(len(df)),
        "intent_accuracy": float(accuracy_score(y_test, predictions)),
        "intent_macro_f1": float(
            f1_score(y_test, predictions, average="macro")
        )
    }

    # Human agreement is calculated ONLY when real A/B annotations exist.
    a = df["annotator_a_intent"].fillna("").astype(str).str.strip()
    b = df["annotator_b_intent"].fillna("").astype(str).str.strip()

    mask = (a != "") & (b != "")

    if mask.sum() >= 2:
        results["human_intent_kappa"] = float(
            cohen_kappa_score(a[mask], b[mask])
        )
    else:
        results["human_intent_kappa"] = None
        print("Human intent kappa: skipped - no real A/B annotations.")

    print("\nRESULTS")
    print(json.dumps(results, indent=2))

    Path("data/primary_eval.json").write_text(
        json.dumps(results, indent=2)
    )

    print("\nSaved: data/primary_eval.json")


if __name__ == "__main__":
    main()
