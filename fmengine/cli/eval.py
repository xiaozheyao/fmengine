import os
from typing import Dict, Any, List
import json


def evaluation_entry(
    hf_ckpt_path: str,
    revision: str="main",
    backend="lm_eval",
    tasks: str = "",
    result_path: str = ".local/eval/results",
    num_fewshot: int = 0,
    backend_args: Dict[str, Any] = {},
):
    try:
        import lm_eval
        from lm_eval.utils import make_table
    except ImportError:
        raise ImportError("Please install lm_eval to use this command.")
    
    lm = lm_eval.models.huggingface.HFLM(pretrained=hf_ckpt_path, revision=revision)
    task_manager = lm_eval.tasks.TaskManager()
    tasks = tasks.split(",")
    results = lm_eval.simple_evaluate(
        model = lm,
        tasks=tasks,
        num_fewshot=num_fewshot,
        task_manager = task_manager
    )
    print(make_table(results))