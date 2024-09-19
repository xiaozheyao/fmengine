from ._base import Callback
from fmengine.evaluator.jsonl_evaluator import JSONLEvaluator
import wandb


class GeneratorValidationCallback(Callback):
    def __init__(self, jsonl_path: str, freq: int) -> None:
        self.evaluator = JSONLEvaluator(jsonl_path)
        self.freq = freq

    def step(self, step, args):
        if step % self.freq == 0:
            model = args["model"]
            tokenizer = args["tokenizer"]
            logger = args["metric_logger"]
            results = self.evaluator.evaluate(model, tokenizer)

            gen_table = wandb.Table(
                columns=["step", "prompt", "output"], data=[[step, res["prompt"], res["output"]] for res in results]
            )

            logger.log({"generation": gen_table}, step, is_table=True)
