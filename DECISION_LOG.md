# Decision Log

1. Selected one brand rather than mixing brands so that reply style and resolution history stay coherent.
2. Started with four intents to keep the taxonomy small enough for reliable human labelling.
3. Included `other` to avoid forcing ambiguous examples into a wrong class.
4. Used inbound customer tweets as primary inputs because the task is to classify incoming customer messages.
5. Used a TF-IDF + Logistic Regression baseline as the simple learned baseline.
6. Included a trivial majority-class baseline to make the comparison explicit.
7. Kept the golden set separate from any development examples.
8. Required two human annotations for the agreement subset instead of treating model agreement as human agreement.
9. Used four judge criteria: intent, escalation, reply grounding, and format/policy compliance.
10. Added deterministic escalation-format validation before the LLM judge.
11. Added a secondary reviewer service over HTTP to match the requested audit architecture.
12. Added bootstrap confidence intervals so the headline metric is not reported as a single unsupported number.
13. Chose not to build a production UI because the assignment evaluates the evaluation system rather than product polish.
14. Chose not to fine-tune a model because the short take-home window makes reproducible evaluation more valuable.
15. Chose a small reproducible sample rather than attempting to process the full multi-million-row dataset.
