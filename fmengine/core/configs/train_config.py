from dataclasses import dataclass
from typing import Union

from fmengine.models.llama.config_llama import LlamaArgs


@dataclass
class TokenizerConfig:
    pretrained: str


@dataclass
class CheckpointConfig:
    ckpt_dir: str


@dataclass
class TrainingConfig:
    gc_freq: int = 1000
    dp_degree: int = 1
    tp_degree: int = 1
    pp_degree: int = 1
    enable_loss_parallel: bool = False
    data_parallel_type: str = "fsdp"
    dump_folder: str = ".local/output"


@dataclass
class TrainJobConfig:
    model: Union[LlamaArgs]
    tokenizer: TokenizerConfig
    checkpoint: CheckpointConfig
    training: TrainingConfig
