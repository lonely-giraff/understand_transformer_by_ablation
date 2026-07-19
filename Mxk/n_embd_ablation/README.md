# n_embd Ablation

**Question**: How small can the embedding dimension be before the model stops learning?

**Experiment**: Train the modern GPT trunk (RoPE, RMSNorm, ReLU^2) at depth=6 on FineWeb text (20M tokens), varying only `model.dim`.

**Arms**:
- dim=384 (baseline — the house rule dim = 64×depth)
- dim=64  (6× smaller)
- dim=16  (24× smaller)

**Result**: On this training budget, **dim=64 converges fastest** (CE 5.85), beating both the larger baseline (dim=384, CE 9.49) and the extreme cut (dim=16, CE 6.42). The baseline with more parameters does not learn faster — smaller models converge better under a fixed token budget.

**Takeaway**: For a given compute/data budget, larger embedding dimensions do not guarantee better results. The optimal embedding size depends on the training budget.
