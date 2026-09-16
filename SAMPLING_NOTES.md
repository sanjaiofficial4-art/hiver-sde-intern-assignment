# Sampling Notes

Brand: AppleSupport (change if a different brand is used).

Sampling plan:
1. Load the Customer Support on Twitter dataset.
2. Keep inbound customer messages for the selected brand.
3. Remove empty/duplicate text.
4. Keep a reproducible random sample of 500–1000 rows for development.
5. Build a 200-row golden set from a separate sample using stratification where possible.
6. Human annotators label intent and escalation independently.
7. At least 50 examples are double-labelled by two humans.
8. Resolve disagreements using the codebook; retain the original annotations for agreement analysis.

Final submission must replace any provisional labels with actual human labels.
