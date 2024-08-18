from typing import Union

from .llama.config_llama import LlamaArgs


def build_model(model_config: Union[LlamaArgs]):
    if model_config.architecture == "llama":
        from .llama.modeling_llama import build_llama_3

        return build_llama_3(model_config)
    else:
        raise NotImplementedError(
            f"Architecture {model_config.architecture} not implemented."
        )
