import os
from typing import Any, Dict, Optional, TYPE_CHECKING

import torch
import wandb

from fmengine.core.configs.train_config import TrainJobConfig

if TYPE_CHECKING:
    from fmengine.core.parallelism.parallel_dims import ParallelDims

wandb.require("core")


def _get_metrics_rank(parallel_dims: "ParallelDims") -> int:
    """
    Returns global rank 0 in non-pipeline-parallel configs, and returns the global
    rank of the 0th rank in the last pipeline stage when pipeline parallelism is enabled.
    """
    if parallel_dims.pp_enabled:
        world_size = parallel_dims.world_size
        pp_size = parallel_dims.pp
        metrics_log_rank = (world_size // pp_size) * (pp_size - 1)
    else:
        metrics_log_rank = 0

    return metrics_log_rank


class MetricLogger:
    def __init__(self, project_name: str, job_config: TrainJobConfig, enable_wb: bool):
        # we don't use job_config's enable_wb, as it might be overwritten by the rank_0_only logic
        if enable_wb:
            self.run = wandb.init(project=project_name)
            config = self.run.config
            config.job = job_config
        else:
            self.run = None

    def log(self, metrics: Dict[str, Any], step: int):
        metrics["step"] = step
        if self.run is not None:
            self.run.log(metrics)

    def close(self):
        pass


def build_metric_logger(job_config: TrainJobConfig, parallel_dims: "ParallelDims", tag: Optional[str] = None):
    """
    parallel_dims is used to determine the rank to log metrics from if 'tb_config.rank_0_only=True'.
    In that case, `_get_metrics_rank` will be used to calculate which rank acts as 'rank 0'. This is
    intended to allow logging from the 0th rank within the last pipeline stage group, in case pipeline
    parallelism is enabled, without forcing logging from all ranks to capture loss information.
    """
    metrics_config = job_config.metrics
    # since we don't have run id, use current minute as the identifier
    enable_wb = metrics_config.enable_wb
    if enable_wb:
        if metrics_config.rank_0_only:
            enable_wb = torch.distributed.get_rank() == _get_metrics_rank(parallel_dims)
        else:
            rank_str = f"rank_{torch.distributed.get_rank()}"
    return MetricLogger(metrics_config.project_name, job_config, enable_wb)
