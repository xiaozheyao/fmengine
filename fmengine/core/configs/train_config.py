from dataclasses import dataclass, field
from typing import Union

from fmengine.models.llama.config_llama import LlamaArgs


@dataclass
class TokenizerConfig:
    pretrained: str


@dataclass
class CheckpointConfig:
    ckpt_dir: str
    keep_latest_k: int = 5
    enable_checkpoint: bool = True


@dataclass
class FP8Config:
    enable_float8_linear: bool = False


@dataclass
class OptimizerConfig:
    name: str = "adam"
    lr: float = 1e-3
    fused: bool = True
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)


@dataclass
class TrainingConfig:
    gc_freq: int = 1000
    dp_degree: int = 1
    tp_degree: int = 1
    pp_degree: int = 1
    warmup_steps: int = 200
    train_steps: int = 1000
    enable_loss_parallel: bool = False
    data_parallel_type: str = "fsdp"
    dump_folder: str = ".local/output"
    enable_async_tensor_parallel: bool = False
    compile: bool = True
    float8: FP8Config = field(default_factory=FP8Config)


@dataclass
class ExperimentalConfig:
    enable_compiled_autograd: bool = False


@dataclass
class TrainJobConfig:
    model: Union["LlamaArgs"]
    tokenizer: TokenizerConfig
    checkpoint: CheckpointConfig
    training: TrainingConfig
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    experimental: ExperimentalConfig = field(default_factory=ExperimentalConfig)
