"""Process-scoped experiment hooks for nanoinfra's standard orchestrator."""

import os


MODE_ENV = "NANOINFRA_SUPERVISION_MODE"
mode = os.environ.get(MODE_ENV)

if mode is not None:
    if os.environ.get("NANOINFRA_EVAL_MODE") == "final":
        import modalities.text as text_modality
        import modalities.text.evaluator as text_evaluator

        from final_token_evaluator import FinalTokenEvaluator

        text_modality.TextEvaluator = FinalTokenEvaluator
        text_evaluator.TextEvaluator = FinalTokenEvaluator

    from core.data.mixed_dataloader import MixedDataLoader
    from core.data.supervision import NextTokenPrediction

    from supervision import LastPositionPrediction, RandomPositionPrediction

    strategies = {
        "all": NextTokenPrediction,
        "last": LastPositionPrediction,
        "projected": LastPositionPrediction,
        "random": RandomPositionPrediction,
    }
    if mode not in strategies:
        raise ValueError(
            f"Unknown {MODE_ENV}={mode!r}; expected one of {sorted(strategies)}"
        )

    original_init = MixedDataLoader.__init__

    def experiment_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        strategy = strategies[mode]
        self.supervision = (
            strategy(seed=int(os.environ.get("NANOINFRA_SUPERVISION_SEED", "42")))
            if mode == "random"
            else strategy()
        )
        print(f"Ablation supervision mode={mode}: {self.supervision.__class__.__name__}")

    MixedDataLoader.__init__ = experiment_init

    if mode == "projected":
        from core.model.system import LMSystem
        from core.training import model_setup
        from core.training.trainer import Trainer

        from projected_head import ProjectedSequenceLMHead, print_projection_summary

        model_setup.LMHead = ProjectedSequenceLMHead
        model_setup.LIGER_AVAILABLE = False

        def projected_estimate_flops(self):
            head = self.head
            head_per_sequence = 6 * head.lm_head.weight.numel()
            head_per_token = head_per_sequence / head.position_logits.numel()
            pooling_per_token = 6 * self.trunk.config.n_embd
            return self.trunk.estimate_flops() + head_per_token + pooling_per_token

        LMSystem.estimate_flops = projected_estimate_flops
        original_train = Trainer.train

        def projected_train(self, *args, **kwargs):
            result = original_train(self, *args, **kwargs)
            print_projection_summary(self.system.head)
            return result

        Trainer.train = projected_train
