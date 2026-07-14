# RoPE frequency study

**What set of rotation frequencies should RoPE use — and does it even matter?**

The base model is **GPT-2 + RoPE** (the [ablation rung](../example_gpt2_vs_modern/) from
the generational A/B: classic GPT-2 body — LayerNorm+bias, tanh-GELU, biased linears,
plain multi-head — with learned absolute position swapped for RoPE). Every experiment
here trains the *same* model, data, budget, and recipe; the **only** thing that varies
is the RoPE frequency schedule — the set of `N = head_dim/2` per-pair rotation rates
`θ_j`. At d6 that is **64 frequencies**, `head_dim=128`.

Standard RoPE picks them geometrically: `period_j = base^(j/N)`, `base=10000` — a
log-uniform sweep of wavelengths over `[1, base]`. (Token wavelength = `2π·period`.)
This study asks what happens if you pick them differently, and — the punchline — finds
that **the schedule barely matters, but for a reason worth knowing**.

## TL;DR — a hierarchy of what matters

![hierarchy](figures/hierarchy.png)

Three tiers, each an order of magnitude apart (d6, seq_len=512, val CE):

| what | measured by | worth |
|---|---|---:|
| **position at all** | no-position → any position | **~2.7 CE** |
| **a multi-frequency *code*** | 1 frequency → 64 frequencies | **~0.2 CE** |
| **the exact schedule** | across wildly different 64-freq schedules | **~0.03 CE** |

The famous `base=10000` geometric law lives entirely in the bottom tier: you can bend
it, rescale it, truncate half of it, or *learn* it, and move val CE by ~0.03. What
actually earns RoPE its keep is one tier up — having a **set of distinct frequencies**
(a positional code that disambiguates position the way a bank of coprime clocks counts
past any single clock's wrap-around), not the particular set.

## The knob

One trunk, [`rope_trunk.py`](rope_trunk.py), whose frequency schedule is set by env
dials (the text orchestrator builds a fixed `GPTConfig` and doesn't thread extra model
config, so per-arm knobs ride the subprocess env that `run.py` sets — **core is
untouched**). The schedule math is [`schedules.py`](schedules.py):

| dial | env | effect |
|---|---|---|
| `base` | `ROPE_BASE` | max period / context reach (θ-scaling / NTK). Sets the top endpoint. |
| `gamma` | `ROPE_GAMMA` | warp exponent on `j/N`. `>1` packs pairs at SHORT periods (local); `<1` at LONG (global). Endpoints pinned; density redistributed. |
| `rotary_pct` | `ROPE_PCT` | fraction of pairs that rotate; the rest get `θ=0` (no position). |
| `single` | `ROPE_SINGLE` | collapse ALL pairs to one frequency (period `single`). |

`gamma=1, base=10000, rotary_pct=1` reproduces standard RoPE bit-for-bit.

## Setup 1 — fixed alternative schedules

![schedules](figures/frequency_schedules.png)
*Left y-axis = period on a **log** scale, so standard RoPE — period = base^(j/N),
exponential in j — is the straight **black** reference line; γ warps it up (γ<1) or down
(γ>1) around it. (A linear-y wavelength plot shows that same geom curve shooting up
steeply at the end — the log axis just straightens it.)*

**(a) Bend / rescale / truncate** — `python run.py` → [`plots.py schedules`](plots.py). Six schedules
that cover the range differently (γ=0.5/2, base=1e3/1e5, rotary=50%) all land within
**0.009 CE** of the geometric baseline (5.55). Even zeroing the position signal on
**half** the channels costs only 0.007 — the positional channels are hugely redundant
at 512 context.

![schedule loss](figures/schedule_loss.png)
The loss curves confirm it (`plots.py schedule_loss`): all six schedules track each other
the whole way and land *inside* the 5-seed noise band (right) — the 0.009 spread is seed
lottery, not schedule.

**(b) The base sweep** — `python run.py base` (single-seed), `python run.py baseseeds`
(5 seeds/base).
![base sweep](figures/base_sweep_seeds.png)
Shrinking `base` compresses every wavelength toward 1. Single-seed it traces a shallow
bowl (span 0.034). With **5 seeds per base** the **descending arm is real**: the mean
climbs monotonically from base≈512 (5.543) up to base=8 (5.572) across low-variance
points (~13 SEM), so shortening the reach below a few hundred tokens genuinely hurts.
What is *not* resolved: the precise optimum (256/512/1024 overlap) and whether base=10000
is truly worse — both confounded by one outlier seed. The whole effect is still tiny
(~0.03 CE): the model's positional *range* needs are modest at 512 context (even a
~50-token reach, base=8, costs only 0.03). This measures the model's **effective
positional range** — a real, if small, effect that lives inside the bottom hierarchy tier.

**(c) Single frequency** — `python run.py single`. Collapsing all 64 pairs to one
frequency recovers **86–93%** of the full positional value (best when the single
wavelength ≈ context, so it makes one clean, alias-free rotation across the window),
but is still **0.19–0.39 CE worse** than 64 frequencies — an order of magnitude more
than the schedule knob. This is the middle tier of the hierarchy. (The core carries a
commented-out fossil of the single-*lowest*-freq variant in
[`gpt.py`](../../core/model/gpt.py#L294-L296).)

**Noise floor** — `python run.py seeds`.
![noise floor](figures/noise_floor.png)
Everything above starts single-seed. Repeating the geometric baseline across 5 seeds gives
**σ≈0.022, range 0.055** (heavy-tailed — one seed at 5.603). So the schedule spread
(0.008 = **0.4σ**) is pure noise: the γ/rotary ranking is seed lottery. The base bowl's
raw spread (0.033 = **1.5σ**) *looks* like noise by extremes too — **but a monotonic trend
across many ordered points carries more significance than its raw spread**, so we seeded it
(above): the descending arm survives, the fine optimum does not. The methodological rule at
this scale: from a *single pairwise* comparison, trust only effects **≳2σ (~0.045)**; a
*consistent trend across a sweep* can be real below that. What clears the bar outright: the
top two hierarchy tiers (position ≈120σ, multi-freq code ≈9σ). The exact schedule sits at
the noise floor — the strongest form of "it barely matters."

## Setup 2 — learnable frequencies

![learned](figures/learned_freqs.png)

[`rope_learnable.py`](rope_learnable.py): make the frequencies a parameter and let
gradient descent choose. `python run.py learn` → [`plots.py learned`](plots.py).

Design (see the discussion notes): a **shared log-residual** `inv_freq = base_freq ·
exp(δ)`, `δ` a `[N]` parameter **init 0** (so step 0 is exactly standard RoPE and it can
only match or improve). Because `δ` is a residual, AdamW's weight decay pulls it *toward
the geometric baseline* — a prior toward standard RoPE, not a pathology — so `δ` rides
the default matrix param-group with **no core change**. `cos/sin` move from a cached
buffer to a per-forward fp32 recompute (bf16 trig on large angles is garbage); gradients
reach `δ` by ordinary autograd because RoPE is applied *outside* the attention kernel
(SDPA vs FlexAttention is irrelevant — this project trains on SDPA).

**Shared-δ result:** val **5.554** — within seed noise of the geometric baseline;
learning the frequencies gains nothing (as Setup 1 predicted). The *profile* is the
payoff: the learned curve barely leaves γ=1 (max `|δ|=0.125` → ≤13% frequency change),
with a weak **systematic** residual that **shortens the longest-wavelength channels**
(positive δ at high j) and lengthens the channels whose wavelength ≈ the context length
(a reproducible dip at j≈30). Suggestive, not confirmed — the signal is at the noise
floor.

**Per-layer** — [`rope_perlayer.py`](rope_perlayer.py), `python run.py perlayer` →
[`plots.py perlayer`](plots.py). Each layer gets its own δ, to ask: do layers
*specialize* (early=local, late=global)?
![per-layer](figures/perlayer_freqs.png)
**Answer: mostly no.** All 6 layers converge to nearly the *same* residual shape (the same
j≈30 dip and high-j rise the shared run found) — they move together, not apart. This is
what the mechanism predicts: every layer already has the full 1→base spectrum and just
uses the subset it needs, so per-layer freedom only lets it *reallocate* resolution, which
buys little. The one weak signal: the **final layer** shortens its longest wavelengths
least (retains long-range → mildly global). Val **5.549**, again within noise.

![setup 2 loss](figures/setup2_loss.png)
Both learnable runs land within **0.65σ** of the 5-seed baseline mean — loss curves
indistinguishable from baseline throughout, finals inside the ±1σ band. **Setup 2's
verdict: learning the frequencies buys nothing on loss at this scale; the geometric
schedule is already in a flat basin, and layers don't want to differ.** (To see whether
the tiny learned profile — the j≈30 dip — is stable or seed noise, it would need seeding
too; and specialization has more to specialize *for* at long context.)

## Files

| file | what |
|---|---|
| `schedules.py` | the RoPE frequency schedules as pure functions of the dials |
| `rope_trunk.py` | `RoPETrunk` — GPT-2+RoPE with a fixed, dial-set schedule (Setup 1) |
| `rope_learnable.py` | `RoPELearnableTrunk` — shared learnable log-residual δ (Setup 2) |
| `rope_perlayer.py` | `RoPEPerLayerTrunk` — per-layer learnable δ (Setup 2) |
| `spec.py` | the arms/sweeps + the recipe (the one place the knobs live) |
| `run.py` | the driver: `run.py [arms\|base\|single\|learn\|perlayer\|seeds\|baseseeds]` |
| `plots.py` | every figure: `plots.py [name\|all]` → `figures/` |
| `results/` | the raw result JSONs (one per sweep) |
| `figures/` | the 7 rendered figures |

## Run it

```bash
python download_data.py   # once: a FineWeb shard

# train the sweeps (each writes results/<sweep>.json; ~2.5 min/arm at d6)
python run.py            # Setup 1a: the 6 schedules
python run.py base        # Setup 1b: base sweep (single-seed)
python run.py single      # Setup 1c: single-frequency
python run.py seeds       # noise floor: geom × 5 seeds
python run.py baseseeds   # base × 5 seeds (the trend test)
python run.py learn       # Setup 2: shared learnable δ
python run.py perlayer    # Setup 2: per-layer learnable δ

python plots.py all       # render every figure into figures/
```

## Figures

`hierarchy` (the headline) · `schedules` (the 6 schedule shapes) · `schedule_loss` (their
loss curves vs noise) · `base_seeds` (the seeded base trend) · `noise` (the noise floor) ·
`learned` (shared-δ profile) · `perlayer` (per-layer profiles) · `setup2` (the Setup-2
loss data). Render one with `python plots.py <name>`, or all with `python plots.py all`.

## What's next

- **Long context.** The whole study is at seq_len=512, where positional *range* barely
  matters — which is *why* the schedule sits at the noise floor. At long context the
  low-frequency channels do real work, so base/schedule (and per-layer specialization)
  should bite above noise. This is the natural extension.
- **Seed the learned profiles.** The Setup-2 δ shape (the j≈30 dip) is a single-seed
  readout; seeding it would tell whether that structure is real or optimization noise.
