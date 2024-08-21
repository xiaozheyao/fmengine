import os
from typing import Dict, Any, List
import subprocess
from fmengine.utilities import logger

def environment_check(use_lm_eval: bool=True):
    try:
        import lm_eval
    except ImportError:
        raise ImportError("lm_eval not found, please install it with `pip install lm_eval @git+https://github.com/EleutherAI/lm-evaluation-harness.git@v0.4.3`")
    from shutil import which
    if which("lm_eval") is None:
        raise FileNotFoundError("lm_eval not found in PATH, please check your installation.")
    return True
        

def evaluation_entry(
        hf_ckpt_path: str,
        backend="lm_eval",
        tasks: str="",
        result_path:str=".local/eval/results",        
        backend_args: Dict[str, Any]={},
    ):
    use_lm_eval = False
    if backend=="lm_eval":
        use_lm_eval = True
    else:
        raise NotImplementedError(f"Backend {backend} not implemented.")
    environment_check(use_lm_eval)
    os.makedirs(result_path, exist_ok=True)
    command_args = f"--model hf --model_args pretrained={hf_ckpt_path},dtype=bfloat16 --tasks {tasks} --batch_size auto:4 --device cuda:0 --output_path {result_path}"
    logger.info(f"Running evaluation with command: lm_eval {command_args}")
    os.system(f"lm_eval {command_args}")
    return True
