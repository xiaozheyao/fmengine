from typing import Union, Dict, Any

from .llama.config_llama import LlamaArgs
from transformers import AutoModelForCausalLM, AutoConfig

def build_model(model_config: Union[LlamaArgs]):
    if model_config.architecture == "llama":
        from .llama.modeling_llama import build_llama_3

        return build_llama_3(model_config)
    else:
        raise NotImplementedError(f"Architecture {model_config.architecture} not implemented.")


def export_to_huggingface(states: Dict[str, Any], model_config: Union[LlamaArgs]) -> (AutoModelForCausalLM, AutoConfig):
    if model_config.architecture == "llama":
        from .llama.interop_llama import to_huggingface

        return to_huggingface(states, model_config)
    else:
        raise NotImplementedError(f"Architecture {model_config.architecture} not implemented.")
