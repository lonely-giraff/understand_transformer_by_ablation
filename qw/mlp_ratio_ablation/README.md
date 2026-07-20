# MLP Expansion Ratio Ablation

Compare MLP expansion ratios in the transformer feed-forward network:
**2x / 4x (baseline) / 6x / 8x**. Same depth (d6), same dim (384),
same data, same budget — only `mlp_ratio` varies.

## Question

Does widening the MLP beyond the standard 4x buy proportional loss
reduction, or do returns diminish?

## Arms

| Arm | MLP Ratio | Non-embedding Params | Trunk |
|-----|-----------|---------------------|-------|
| ratio_2x | 2x | 7.08M | `GPT_MLP2x` |
| ratio_4x | 4x (baseline) | 10.62M | reference GPT |
| ratio_6x | 6x | 14.16M | `GPT_MLP6x` |
| ratio_8x | 8x | 17.70M | `GPT_MLP8x` |

## Finding (200M tokens, proper train/val split)

| Arm | Best Val CE | Δ vs 4x |
|-----|-----------|---------|
| 2x | 5.04 | +0.10 (worse) |
| 4x | 4.94 | baseline |
| 6x | 4.91 | −0.03 |
| **8x** | **4.89** | **−0.05** |

With 200M tokens, 8x is the winner — but diminishing returns are
severe: going from 4x→6x→8x costs 3.5M params each step for only
0.03–0.02 CE. The standard 4x is within 0.05 CE of the optimum.

At smaller budgets (20M tokens), 6x and 8x underperform due to
insufficient data — larger MLPs need more tokens to converge.

## Run

```bash
pip install -r requirements.txt
python download_data.py
python -m modalities.text.train_tokenizer

cd qw/mlp_ratio_ablation
python run.py    # trains all 4 arms
python plot.py   # -> mlp_ratio_ablation.png
```

## Credit

Built on [nanoinfra](https://github.com/suning-git/nanoinfra).
Zero core edits — each ratio is a thin GPT subclass via
`model.trunk_class`.
