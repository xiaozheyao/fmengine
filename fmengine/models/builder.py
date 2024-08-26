from typing import Union, Dict, Any, Tuple, TYPE_CHECKING

from .llama.config_llama import LlamaArgs
from transformers import AutoModelForCausalLM, AutoConfig

if TYPE_CHECKING:
    from fmengine.core.configs.train_config import AutoOptimizationFlags

def build_model(model_config: Union[LlamaArgs], ao_flags: "AutoOptimizationFlags"):
    if model_config.architecture == "llama":
        from .llama.modeling_llama import build_llama_3

        return build_llama_3(model_config, ao_flags)
    else:
        raise NotImplementedError(f"Architecture {model_config.architecture} not implemented.")


def export_to_huggingface(
    states: Dict[str, Any], export_dtype: str, model_config: Union[LlamaArgs]
) -> Tuple[AutoModelForCausalLM, AutoConfig]:
    if model_config.architecture == "llama":
        from .llama.interop_llama import to_huggingface

        return to_huggingface(states, export_dtype, model_config)
    else:
        raise NotImplementedError(f"Architecture {model_config.architecture} not implemented.")


def import_from_huggingface(model_arch: str, pretrained_model_id_or_path: str, load_dtype: str):
    if model_arch == "llama":
        from .llama.interop_llama import from_huggingface

        return from_huggingface(pretrained_model_id_or_path, load_dtype)
    else:
        raise NotImplementedError(f"Architecture {model_arch} not implemented.")
