import contextlib
import os
import time

import humanize
import torch
from torch.distributed.elastic.multiprocessing.errors import record
from torch.fx import GraphModule

from fmengine.core.checkpoint import CheckpointManager, TrainState
from fmengine.core.configs.train_config import TrainJobConfig
from fmengine.core.nn import build_lr_scheduler, build_optimizer
from fmengine.core.nn.loss import cross_entropy_loss
from fmengine.core.parallelism.distributed import init_distributed
from fmengine.core.parallelism.parallel_dims import ParallelDims
from fmengine.datasets import build_hf_data_loader
from fmengine.datasets.tokenizer import build_tokenizer
from fmengine.models.builder import build_model
from fmengine.models.llama.modeling_llama import parallelize_llama
from fmengine.models.utils import get_num_params
from fmengine.utilities import (
    GarbageCollection,
    build_gpu_memory_monitor,
    get_peak_flops,
    logger,
)


def get_train_context(enable_loss_parallel: bool, enable_compiled_autograd: bool):
    @contextlib.contextmanager
    def context():
        with contextlib.ExitStack() as stack:
            if enable_loss_parallel:
                stack.enter_context(torch.distributed.tensor.parallel.loss_parallel())
            if enable_compiled_autograd:
                stack.enter_context(
                    torch._dynamo.utils.maybe_enable_compiled_autograd(True)
                )
            yield

    return context


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
    init_distributed(dump_folder=job_config.training.dump_folder)
    world_mesh = parallel_dims.build_mesh(device_type="cuda")
    print(world_mesh)
    gpu_memory_monitor = build_gpu_memory_monitor()
    gpu_peak_flops = get_peak_flops(gpu_memory_monitor.device_name)
    if parallel_dims.dp_enabled:
        dp_mesh = world_mesh["dp"]
        dp_degree, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
    else:
        dp_degree, dp_rank = 1, 0
    # build model
    with torch.device("meta"):
        model = build_model(job_config.model)
    # todo(xiaozhe): handle fp8 here
    print(model)
    # model stats
    model_param_count = get_num_params(model)
    logger.info(f"Model has {humanize.intword(model_param_count)} parameters")
    # todo(xiaozhe): pipeline parallelism enabled
    parallelize_llama(
        model, world_mesh, parallel_dims, train_config=job_config.training
    )
    init_device = "cuda"
    model.to_empty(device=init_device)
    model_parts = [model]

    for mod in model_parts:
        # skip traced modules since we do not define init_weights in the traced module
        if isinstance(mod, GraphModule):
            continue
        mod.init_weights()
        mod.train()
    logger.info("Model initialized")
    gpu_mem_stats = gpu_memory_monitor.get_peak_stats()
    logger.info(
        f"GPU memory usage for model: "
        f"{gpu_mem_stats.max_reserved_gib:.2f}GiB"
        f"({gpu_mem_stats.max_reserved_pct:.2f}%)"
    )
    # Build optimizer and scheduler
    optimizer = build_optimizer(model_parts, job_config.optimizer)
    scheduler = build_lr_scheduler(optimizer.optimizers, job_config)
    tokenizer = build_tokenizer(
        job_config.tokenizer.tokenizer_type, job_config.tokenizer.tokenizer_name_or_path
    )
    # build dataloader
    data_loader = build_hf_data_loader(
        job_config.dataset.name,
        job_config.dataset.path,
        tokenizer,
        job_config.dataset.batch_size,
        job_config.dataset.seq_len,
        dp_degree,
        dp_rank,
    )
    train_state = TrainState()

    # load initial checkpoint
    checkpoint = CheckpointManager(
        dataloader=data_loader,
        model_parts=model_parts,
        optimizers=optimizer.optimizers,
        lr_schedulers=scheduler.schedulers,
        states={"train_state": train_state},
        ckpt_config=job_config.checkpoint,
    )

    train_context = get_train_context(
        parallel_dims.loss_parallel_enabled,
        job_config.experimental.enable_compiled_autograd,
    )
    logger.info(f"training starts at {time.time()}")
    time.sleep(10000)

    torch.distributed.destroy_process_group()
