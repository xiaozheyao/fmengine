from dataclasses import dataclass, field
from typing import Union

from fmengine.models.llama.config_llama import LlamaArgs


@dataclass
class TokenizerConfig:
    pretrained: str


@dataclass
class CheckpointConfig:
    ckpt_dir: str

@dataclass
class FP8Config:
    enable_float8_linear: bool = False

@dataclass
class TrainingConfig:
    gc_freq: int = 1000
    dp_degree: int = 1
    tp_degree: int = 1
    pp_degree: int = 1
    enable_loss_parallel: bool = False
    data_parallel_type: str = "fsdp"
    dump_folder: str = ".local/output"
    enable_async_tensor_parallel: bool = False
    compile: bool = True
    # use default_factory
    float8: FP8Config = field(default_factory=FP8Config)
    
@dataclass
class TrainJobConfig:
    model: Union[LlamaArgs]
    tokenizer: TokenizerConfig
    checkpoint: CheckpointConfig
    training: TrainingConfig
