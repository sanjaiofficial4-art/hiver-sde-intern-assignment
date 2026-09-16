# Hiver SDE Intern Take-Home Report

## 1. Problem framing
I selected **AppleSupport** from Customer Support on Twitter. The goal is a small support agent that classifies incoming customer messages, drafts grounded replies based on historical support interactions, and decides whether to auto-handle or escalate with a reason.

**Good means:** correct intent classification, safe escalation decisions, and replies that do not invent account-specific facts.

**Chosen not to build:** production UI, authentication, real-time monitoring, model fine-tuning, multilingual support, and full-dataset processing.

## 2. Data and annotation
Dataset: Customer Support on Twitter. Development sample: ___ rows. Golden set: ___ hand-labelled rows. Sampling seed: 42.

Intents:
- account_issue
- billing_refund
- technical_support
- other

Escalation policy: see CODEBOOK.md.

## 3. Baselines and results
Fill this table from `results/eval_results.json`.

| System | Intent Accuracy | Macro F1 | Escalation Precision | Escalation Recall |
|---|---:|---:|---:|---:|
| Trivial | ___ | ___ | ___ | ___ |
| Keyword | ___ | ___ | ___ | ___ |
| TF-IDF + Logistic Regression | ___ | ___ | ___ | ___ |
| LLM agent | ___ | ___ | ___ | ___ |

Do not invent values.

## 4. Judge and human agreement
Judge criteria:
1. classification correctness
2. escalation correctness
3. reply grounding
4. format/policy compliance

Report Cohen's kappa for at least 50 double-labelled examples. Also report judge-vs-human kappa per criterion where available.

## 5. Confidence interval
Headline metric: ___
95% bootstrap CI: [___, ___]

## 6. Top five failure modes
Use real examples from the evaluation set.

1. ___ — example: ___ — hypothesis: ___
2. ___ — example: ___ — hypothesis: ___
3. ___ — example: ___ — hypothesis: ___
4. ___ — example: ___ — hypothesis: ___
5. ___ — example: ___ — hypothesis: ___

## 7. What is misleading about my headline number?
The headline metric is measured on a small, human-labelled sample rather than the full production distribution. It may also be affected by class balance, repeated wording, sampling choices, and correlation between examples. The final version must identify one concrete mechanism observed in this dataset rather than making only a generic caveat.

## 8. One more week
I would improve annotation quality, add an adversarial/OOD evaluation set, improve retrieval of historical resolutions, calibrate escalation thresholds, and add monitoring for distribution shift.

## 9. Conclusion
The final claim should be limited to what the evaluation supports. The main objective is a reproducible evaluation and audit pipeline, not a claim of production readiness.
