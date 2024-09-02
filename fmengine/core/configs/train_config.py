from dataclasses import dataclass, field
from typing import Optional, Union, TYPE_CHECKING
from fmengine.models.llama.config_llama import LlamaArgs

@dataclass
class TokenizerConfig:
    tokenizer_type: str
    tokenizer_name_or_path: str


@dataclass
class CheckpointConfig:
    ckpt_dir: str
    keep_latest_k: int = 5
    enable_checkpoint: bool = True
    interval: int = 1000
    interval_type: str = "steps"
    model_weights_only: bool = False
    export_dtype: str = "bfloat16"
    async_mode: str = "async"
    create_seed_checkpoint: bool = False
    finetuned_from: Optional[str] = None


@dataclass
class FP8Config:
    enable_float8_linear: bool = False


@dataclass
class OptimizerConfig:
    name: str = "adamw"
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
    accumulate_steps: int = 1
    enable_loss_parallel: bool = False
    data_parallel_type: str = "fsdp"
    dump_folder: str = ".local/output"
    enable_async_tensor_parallel: bool = False
    compile: bool = True
    max_norm: float = 1.0
    float8: FP8Config = field(default_factory=FP8Config)
    ac_mode: str = "none"
    cpu_offload: bool = False
    selective_ac_option: str = "2"
    mixed_precision_param: str = "bfloat16"
    mixed_precision_reduce: str = "float32"
    experimental_enable_compiled_autograd: bool = False


@dataclass
class ExperimentalConfig:
    enable_compiled_autograd: bool = False


@dataclass
class DatasetConfig:
    name: str = "c4"
    stream: bool = True
    path: Optional[str] = None
    batch_size: int = 4
    seq_len: int = 2048


@dataclass
class MetricConfig:
    enable_wb: bool = False
    project_name: str = "fmengine"
    project_group: str = "fmengine-dev"
    project_id: Optional[str] = None
    rank_0_only: bool = True
    log_freq: int = 10


@dataclass
class ProfilingConfig:
    enable_profiling: bool = True
    enable_memory_snapshot: bool = False
    save_traces_folder: str = ".local/traces"
    save_memory_snapshot_folder: str = ".local/memory_snapshots"
    profile_freq: int = 10


@dataclass
class TrainJobConfig:
    model: Union[LlamaArgs]
    tokenizer: TokenizerConfig
    checkpoint: CheckpointConfig
    training: TrainingConfig
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    experimental: ExperimentalConfig = field(default_factory=ExperimentalConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    metrics: MetricConfig = field(default_factory=MetricConfig)
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)


@dataclass
class AutoOptimizationFlags:
    use_transformer_engine: bool = False
