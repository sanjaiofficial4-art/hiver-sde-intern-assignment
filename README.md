# Hiver SDE Intern Take-Home — Customer Support AI + Secondary Reviewer

## Goal
Build a small customer-support agent for one brand from the Customer Support on Twitter dataset, then prove whether it is trustworthy using a hand-labelled golden set, two baselines, automated metrics, an LLM judge, human agreement, failure analysis and a secondary HTTP reviewer.

The assignment says the proof is more important than the system.

## Dataset
Primary dataset: Kaggle `thoughtvector/customer-support-on-twitter`.
Dataset fields include `tweet_id`, `author_id`, `inbound`, `created_at`, `text`, `response_tweet_id`, and `in_response_to_tweet_id`.

For a fast reproducible run, use a 500–1000 row brand-specific sample. The full dataset is not required.

## Quick start
```bash
pip install -r requirements.txt
python run_pipeline.py --brand AppleSupport --sample
python evaluate.py --golden_set data/golden_200.csv
python reviewer_service.py
# in another terminal:
python send_for_review.py
```

For Google Colab, run the supplied notebook or execute the same Python files from `/content/hiver_takehome_fast`.

## Important honesty rule
`data/golden_200.csv` must contain human labels before final submission. Do NOT present automatically generated labels as hand labels. The notebook creates an annotation sheet so two annotators can independently label 200 examples.

## Intents
Start with four intents and revise them after inspecting the selected brand:
- account_issue
- billing_refund
- technical_support
- other

If the data clearly suggests different intents, update `CODEBOOK.md` and the code.

## Deliverables
- runnable pipeline
- 150–250 hand-labelled golden examples
- two baselines: trivial + TF-IDF logistic regression
- automated metrics
- LLM judge with four criteria
- human agreement / Cohen's kappa
- secondary reviewer HTTP service
- final audit JSON
- report
- decision log
