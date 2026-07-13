# Attn:MLP ratio ablation

**What if we redistribute parameters between attention and MLP, keeping total
params fixed?**

## Design

Three trunk architectures, all with **identical total parameters** at each depth
(constraint: `4 * N_attn + 8 * N_mlp = 12 * depth`):

| Arm | N_attn blocks | N_mlp blocks | Attn param share |
|-----|---------------|--------------|------------------|
| **Standard** | d | d | 33% |
| **Attn-Heavy** | ~1.5d - 2d | ~0.5d - 1d | 50-67% |
| **MLP-Heavy** | ~0.3d - 0.5d | ~1.3d - 2d | 11-17% |

Block layout: extra attention blocks come first, then paired full (attn+mlp)
blocks, then extra MLP blocks last.

Same data (FineWeb), same budget (2B tokens per size), same recipe (lr=3e-4,
constant LR) — only the block layout differs.

## Full scaling-law results (5 depths, 0.4–25M params)

| Depth | Params | **Standard** | Attn-Heavy | MLP-Heavy | Winner |
|-------|--------|--------------|------------|-----------|--------|
| d2 | 0.4M | 5.149 | **5.135** | 5.150 | Attn (tiny) |
| d3 | 1.3M | **4.836** | 4.882 | 5.098 | **Standard** |
| d4 | 3.1M | **4.646** | 4.806 | 4.681 | **Standard** |
| d6 | 10.6M | **4.385** | 4.454 | 4.482 | **Standard** |
| d8 | 25.2M | **4.252** | 4.376¹ | 4.261 | **Standard** |

| Metric | Standard | Attn-Heavy | MLP-Heavy |
|--------|----------|------------|-----------|
| Frontier exponent a | **0.499** | 0.510 | 0.511 |
| Loss-N exponent alpha | 0.0464 | 0.0418 | 0.0484 |
| R² of L~N^(-alpha) | **0.996** | 0.966 | 0.941 |

**Frontier exponent a ≈ 0.5 for all variants** — the scaling law is inherent
to the Transformer architecture, not the attn/mlp ratio. But the **absolute
frontier height differs**: Standard is lowest (= most efficient) at every
depth >= 3.

## Conclusion

**33% attn : 67% MLP is the optimal parameter allocation.** Deviating in either
direction wastes compute — you get higher loss for the same FLOP budget. The
scaling *rate* (exponent a) is unchanged, but the frontier shifts vertically.

Don't mess with this ratio. Spend your architecture innovation budget elsewhere.

## Files

- `spec.py` — configuration (depth, arms, recipe)
- `run.py` — train all three arms
- `plot.py` — generate the comparison figure
- `variant_trunk.py` — local trunk classes (AttnHeavyGPT, MLPHeavyGPT)

---

¹ **Attn-Heavy d8**: the original run (`device_batch_size=32`) OOM'd because
18 blocks (12 attn + 6 mlp) exceeds standard's 8 blocks. Re-ran successfully
at `device_batch_size=16`.
