from torch import nn
from torch.distributed import DeviceMesh
from typing import Union, Dict, Any, Tuple, TYPE_CHECKING, Optional
from transformers import AutoModelForCausalLM, AutoConfig

if TYPE_CHECKING:
    from fmengine.core.parallelism.parallel_dims import ParallelDims
    from fmengine.core.configs.train_config import TrainingConfig
    from .llama.config_llama import LlamaArgs
    from fmengine.core.configs.train_config import AutoOptimizationFlags


def build_model(model_config: Union["LlamaArgs"], ao_flags: Optional["AutoOptimizationFlags"]):
    if model_config.architecture == "llama":
        from .llama.modeling_llama import build_llama_3

        return build_llama_3(model_config, ao_flags)
    else:
        raise NotImplementedError(f"Architecture {model_config.architecture} not implemented.")


def export_to_huggingface(
    states: Dict[str, Any], export_dtype: str, model_config: Union["LlamaArgs"]
) -> Tuple[AutoModelForCausalLM, AutoConfig]:
    if model_config.architecture == "llama":
        from .llama.interop_llama import to_huggingface

        return to_huggingface(states, export_dtype, model_config)
    else:
        raise NotImplementedError(f"Architecture {model_config.architecture} not implemented.")


def import_from_huggingface(
    model_arch: str, pretrained_model_id_or_path: str, load_dtype: str, ao_flags: "AutoOptimizationFlags"
):
    if model_arch == "llama":
        from .llama.interop_llama import from_huggingface

        return from_huggingface(pretrained_model_id_or_path, load_dtype, ao_flags)
    else:
        raise NotImplementedError(f"Architecture {model_arch} not implemented.")


def parallelize_model(
    model_arch: str,
    model: nn.Module,
    world_mesh: DeviceMesh,
    parallel_dims: "ParallelDims",
    train_config: "TrainingConfig",
):
    if model_arch == "llama":
        from .llama.modeling_llama import parallelize_llama

        return parallelize_llama(model, world_mesh, parallel_dims, train_config)
    else:
        raise NotImplementedError(f"Architecture {model_arch} not implemented.")
