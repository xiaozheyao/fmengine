import os

import torch
from torch.distributed.elastic.multiprocessing.errors import record

from fmengine.core.configs.train_config import TrainJobConfig
from fmengine.core.parallelism.distributed import init_distributed
from fmengine.core.parallelism.parallel_dims import ParallelDims
from fmengine.models.builder import build_model
from fmengine.utilities import (GarbageCollection, build_gpu_memory_monitor,
                                get_peak_flops)


@record
def train_entry(job_config: TrainJobConfig):
    gc_handler = GarbageCollection()
    world_size = int(os.environ["WORLD_SIZE"])
    parallel_dims = ParallelDims(
        dp=job_config.training.dp_degree,
        tp=job_config.training.tp_degree,
        pp=job_config.training.pp_degree,
        world_size=world_size,
        enable_loss_parallel=job_config.training.enable_loss_parallel,
        dp_type=job_config.training.data_parallel_type,
    )
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    print(f"Training on device: {device}")
    init_distributed(dump_folder=job_config.training.dump_folder)
    world_mesh = parallel_dims.build_mesh(device_type="cuda")
    gpu_memory_monitor = build_gpu_memory_monitor()
    gpu_peak_flops = get_peak_flops(gpu_memory_monitor.device_name)
    if parallel_dims.dp_enabled:
        dp_mesh = world_mesh["dp"]
        dp_degree, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
    else:
        dp_degree, dp_rank = 1, 0
    model = build_model(job_config.model)
    torch.distributed.destroy_process_group()
    # build model
