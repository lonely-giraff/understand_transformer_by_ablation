"""Train the comparison arms with a shared final-token evaluator."""

import argparse
import json

import spec
from run import HERE, RESULTS, eval_schedule, run_arm

FINAL_RESULT = RESULTS / "shared_final_token_d12_50m.json"
FINAL_OUTPUTS = HERE / "outputs" / f"{spec.RUN_TAG}_final"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--smoke-mode",
        choices=[mode for _, mode in spec.FINAL_TOKEN_ARMS],
        default="all",
    )
    args = parser.parse_args()

    max_steps = 3 if args.smoke else int(spec.MAX_TOKENS // spec.TBS)
    eval_at = [2] if args.smoke else eval_schedule(max_steps)
    selected = (
        [(f"smoke_{args.smoke_mode}", args.smoke_mode)]
        if args.smoke
        else spec.FINAL_TOKEN_ARMS
    )
    output_dir = FINAL_OUTPUTS / "smoke" if args.smoke else FINAL_OUTPUTS
    arms = [
        run_arm(
            label,
            mode,
            max_steps,
            eval_at,
            primary_metric="final_ce",
            min_evals=1 if args.smoke else 3,
            warmup_steps=1 if args.smoke else spec.WARMUP_STEPS,
            output_dir=output_dir,
            extra_env={
                "NANOINFRA_EVAL_MODE": "final",
                "NANOINFRA_FINAL_EVAL_SEQUENCES": "64" if args.smoke else "2048",
            },
        )
        for label, mode in selected
    ]
    if args.smoke:
        return

    payload = {
        "benchmark": "fixed_context_final_token",
        "depth": spec.DEPTH,
        "context_len": spec.SEQ_LEN - 1,
        "eval_sequences": 2048,
        "max_tokens": spec.MAX_TOKENS,
        "max_steps": max_steps,
        "seed": spec.SEED,
        "arms": arms,
    }
    RESULTS.mkdir(exist_ok=True)
    FINAL_RESULT.write_text(json.dumps(payload, indent=2))
    print(f"WROTE {FINAL_RESULT}")


if __name__ == "__main__":
    main()
