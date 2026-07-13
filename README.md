# understand transformers by ablation

A gallery of controlled transformer experiments: remove or swap ONE thing, train it against a baseline, read the two curves. Each experiment is a **self-contained folder** built on [nanoinfra](https://github.com/suning-git/nanoinfra).

## Run an experiment

```bash
git clone <this repo> && cd <this repo>
python -m venv .venv && . .venv/bin/activate   # a fresh virtualenv
pip install -r requirements.txt                # installs nanoinfra (the framework) as a library
python download_data.py                        # fetch a couple of FineWeb shards

cd suning/example_gpt2_vs_modern               # any experiment folder
python run.py                                  # train the arms (needs a CUDA GPU)
python plot.py                                 # -> the figure
```

Each folder names its trunk as a LOCAL module and finds nanoinfra automatically, so it runs from wherever it sits — nothing to place under nanoinfra by hand. `download_data.py` puts FineWeb under `./outputs`; override the location with `NANOINFRA_BASE_DIR`.

## Contribute your own

Fork this repo, add a folder `<yourname>/<your_experiment>/` (copy an existing one as a template — `spec.py` + `run.py` + `plot.py` + a local trunk module + a `README.md` with your finding), and open a pull request. One subdirectory per contributor: your PR only touches your own folder, so it never conflicts with anyone else.

---

### [example: GPT-2 vs a modern architecture](suning/example_gpt2_vs_modern/)
`suning/example_gpt2_vs_modern` · suning



![example: GPT-2 vs a modern architecture](suning/example_gpt2_vs_modern/gpt2_vs_modern.png)

### [example: residual-connection ablation](suning/example_residual_ablation/)
`suning/example_residual_ablation` · suning

Remove the

![example: residual-connection ablation](suning/example_residual_ablation/residual_ablation.png)

### [RoPE frequency study](suning/rope_study/)
`suning/rope_study` · suning



![RoPE frequency study](suning/rope_study/figures/base_sweep_seeds.png)

### [RoPE ablation](jiayq/rope_ablation/)
`jiayq/rope_ablation` · jiayq

Remove **rotary position embeddings (RoPE)** from the transformer's attention. The no-RoPE model is
not simply worse — it descends to CE 6.13 alongside the baseline, then **collapses** back to random
(CE 10.40) by the end of training. A three-phase crash: surface learning → conflict accumulation →
catastrophic forgetting. **Conclusion: RoPE is load-bearing infrastructure, not an optimization.**

![RoPE ablation](jiayq/rope_ablation/rope_ablation.png)

