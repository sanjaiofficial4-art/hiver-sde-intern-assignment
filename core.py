import json, os, re, random
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split

INTENTS = ["account_issue", "billing_refund", "technical_support", "other"]

KEYWORDS = {
    "account_issue": ["login","log in","password","locked","account","sign in","verification","verify"],
    "billing_refund": ["charge","charged","refund","payment","billing","subscription","money"],
    "technical_support": ["error","crash","broken","not working","issue","bug","app","website","update"],
}

ESCALATE_RE = re.compile(r"^(Auto-handle|Escalate: .+)$", re.I)

def heuristic_escalation(text):
    t = text.lower()
    risk = ["fraud","unauthorised","unauthorized","hacked","stolen","privacy","legal","safety","unknown charge","scam"]
    return any(x in t for x in risk)

def keyword_intent(text):
    t = text.lower()
    scores = {k: sum(x in t for x in v) for k,v in KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] else "other"

def trivial_predict(texts):
    return ["other"] * len(texts)

def train_simple(train_df):
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=30000)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])
    model.fit(train_df["text"], train_df["intent"])
    return model

def evaluate_predictions(y_true, y_pred, esc_true=None, esc_pred=None):
    out = {
        "intent_accuracy": float(accuracy_score(y_true, y_pred)),
        "intent_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    }
    if esc_true is not None and esc_pred is not None:
        out["escalation_precision"] = float(precision_score(esc_true, esc_pred, zero_division=0))
        out["escalation_recall"] = float(recall_score(esc_true, esc_pred, zero_division=0))
    return out

def bootstrap_accuracy(y_true, y_pred, n=1000, seed=42):
    rng = random.Random(seed)
    vals=[]
    idx=list(range(len(y_true)))
    for _ in range(n):
        s=[rng.choice(idx) for _ in idx]
        vals.append(sum(y_true[i]==y_pred[i] for i in s)/len(s))
    vals.sort()
    return {"estimate": accuracy_score(y_true,y_pred),
            "lower": vals[int(0.025*n)],
            "upper": vals[int(0.975*n)]}
