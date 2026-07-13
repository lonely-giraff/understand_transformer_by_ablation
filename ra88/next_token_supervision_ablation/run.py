"""Train the standard supervision arms and collect validation curves."""

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np

import spec

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RUN_OUTPUTS = HERE / "outputs" / spec.RUN_TAG
STANDARD_RESULT = RESULTS / "supervision_d12_50m.json"

POOL_RE = re.compile(r"PROJECTED_POOL_SUMMARY\s+(\{.*\})")
STEP_RE = re.compile(r"Step\s+(\d+)")
VAL_RE = re.compile(r"val/([a-z_]+):\s+([\d.]+)")
TRAIN_RE = re.compile(
    r"Step\s+\d+/\d+.*?dt:\s+([\d.]+)ms\s+\|\s+tok/s:\s+([\d,]+)"
)


def _nanoinfra_checkout(start):
    """Return a source checkout containing core/ and modalities/, if present."""
    path = start
    while path != path.parent:
        if (path / "core").is_dir() and (path / "modalities").is_dir():
            return path
        path = path.parent
    return None


def _subprocess_env(mode, extra_env=None):
    checkout = _nanoinfra_checkout(HERE)
    repo_root = HERE.parents[1]
    python_path = [str(HERE)] + ([str(checkout)] if checkout else [])
    if os.environ.get("PYTHONPATH"):
        python_path.append(os.environ["PYTHONPATH"])
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(python_path),
        "NANOINFRA_SUPERVISION_MODE": mode,
        "NANOINFRA_SUPERVISION_SEED": str(spec.SEED),
        "NANOINFRA_SEQUENCE_LEN": str(spec.SEQ_LEN),
    }
    env.setdefault(
        "NANOINFRA_BASE_DIR",
        str(checkout / "outputs" if checkout else repo_root / "outputs"),
    )
    if extra_env:
        env.update(extra_env)
    return env, str(checkout or HERE)


def eval_schedule(max_steps, n=spec.N_EVALS, first=5):
    values = np.unique(np.round(np.logspace(np.log10(first), np.log10(max_steps), n)))
    return [int(value) for value in values]


def _parse_trajectory(output, primary_metric):
    trajectory = []
    for line in output.splitlines():
        step_match = STEP_RE.search(line)
        metrics = {name: float(value) for name, value in VAL_RE.findall(line)}
        if step_match and primary_metric in metrics:
            point = {
                "step": int(step_match.group(1)),
                "val": metrics[primary_metric],
            }
            point.update(metrics)
            trajectory.append(point)
    return trajectory


def run_arm(
    label,
    mode,
    max_steps,
    eval_at,
    primary_metric="text_ce",
    min_evals=3,
    warmup_steps=spec.WARMUP_STEPS,
    output_dir=RUN_OUTPUTS,
    extra_env=None,
):
    overrides = spec.train_overrides(max_steps, eval_at, warmup_steps)
    env, cwd = _subprocess_env(mode, extra_env)
    print(f"[run ] {label}: mode={mode}, d{spec.DEPTH}, {max_steps} steps", flush=True)
    process = subprocess.run(
        [sys.executable, "-u", "-m", spec.ORCHESTRATOR, *overrides],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
    )
    output = process.stdout + "\n" + process.stderr
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{label}.log").write_text(output)
    trajectory = _parse_trajectory(output, primary_metric)
    if process.returncode != 0 or len(trajectory) < min_evals:
        raise SystemExit(
            f"arm {label} FAILED (rc={process.returncode}, {len(trajectory)} evals); "
            f"see {output_dir / f'{label}.log'}\n{output[-3000:]}"
        )

    timing_samples = [
        {
            "step_ms": float(step_ms),
            "tokens_per_second": int(tokens_per_second.replace(",", "")),
        }
        for step_ms, tokens_per_second in TRAIN_RE.findall(output)
    ]
    steady_samples = timing_samples[1:]
    timing = {
        "compile_step_ms": timing_samples[0]["step_ms"],
        "steady_step_ms_median": statistics.median(
            sample["step_ms"] for sample in steady_samples
        ),
        "steady_tokens_per_second_median": statistics.median(
            sample["tokens_per_second"] for sample in steady_samples
        ),
        "steady_samples": len(steady_samples),
    }
    print(
        f"[done] {label}: {len(trajectory)} evals, "
        f"val {trajectory[0]['val']:.3f} -> {trajectory[-1]['val']:.3f}",
        flush=True,
    )
    result = {"arm": label, "mode": mode, "trajectory": trajectory, "timing": timing}
    pool_match = POOL_RE.search(output)
    if pool_match:
        result["projection"] = json.loads(pool_match.group(1))
    return result


def _payload(arms, benchmark):
    max_steps = int(spec.MAX_TOKENS // spec.TBS)
    return {
        "benchmark": benchmark,
        "depth": spec.DEPTH,
        "sequence_len": spec.SEQ_LEN,
        "max_tokens": spec.MAX_TOKENS,
        "max_steps": max_steps,
        "seed": spec.SEED,
        "arms": arms,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--smoke-mode",
        choices=[mode for _, mode in spec.ALL_ARMS],
        default="all",
    )
    parser.add_argument(
        "--only-arm",
        choices=[label for label, _ in spec.SUPERVISION_ARMS],
    )
    args = parser.parse_args()

    if args.smoke:
        run_arm(
            f"smoke_{args.smoke_mode}",
            args.smoke_mode,
            max_steps=3,
            eval_at=[2],
            min_evals=1,
            warmup_steps=1,
            output_dir=RUN_OUTPUTS / "smoke",
        )
        return

    max_steps = int(spec.MAX_TOKENS // spec.TBS)
    steps = eval_schedule(max_steps)
    selected = [
        (label, mode)
        for label, mode in spec.SUPERVISION_ARMS
        if args.only_arm is None or label == args.only_arm
    ]
    arms = [run_arm(label, mode, max_steps, steps) for label, mode in selected]
    payload = _payload(arms, "standard_all_position_ce")

    if args.only_arm and STANDARD_RESULT.exists():
        existing = json.loads(STANDARD_RESULT.read_text())
        by_label = {arm["arm"]: arm for arm in existing["arms"]}
        by_label.update({arm["arm"]: arm for arm in arms})
        payload["arms"] = [
            by_label[label]
            for label, _ in spec.SUPERVISION_ARMS
            if label in by_label
        ]

    RESULTS.mkdir(exist_ok=True)
    STANDARD_RESULT.write_text(json.dumps(payload, indent=2))
    print(f"WROTE {STANDARD_RESULT}")


if __name__ == "__main__":
    main()
