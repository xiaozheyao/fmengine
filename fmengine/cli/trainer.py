import contextlib
import os
import time

import humanize
import torch
from torch.distributed.elastic.multiprocessing.errors import record
from torch.fx import GraphModule
from datetime import timedelta

from fmengine.core.checkpoint import CheckpointManager, TrainState
from fmengine.core.configs.train_config import TrainJobConfig
from fmengine.core.configs import TORCH_DTYPE_MAP
from fmengine.core.nn import cross_entropy_loss, build_lr_scheduler, build_optimizer
from fmengine.core.parallelism.distributed import init_distributed
from fmengine.core.parallelism.parallel_dims import ParallelDims
from fmengine.data import build_hf_data_loader
from fmengine.data.tokenizer import build_tokenizer
from fmengine.models.builder import build_model, parallelize_model
from fmengine.models.utils import get_num_params
from fmengine.callbacks import build_callbacks

from fmengine.utilities import (
    GarbageCollection,
    build_gpu_memory_monitor,
    build_metric_logger,
    get_peak_flops,
    logger,
    maybe_enable_memory_snapshot,
    maybe_enable_profiling,
    dist_mean,
    dist_max,
    get_num_flop_per_token,
    Color,
    set_pg_timeouts,
    auto_patch,
    set_default_dtype,
)


def get_train_context(
    enable_loss_parallel: bool,
    enable_compiled_autograd: bool,
    enable_mixed_precision: bool,
    mixed_precision_param_dtype: str,
):
    @contextlib.contextmanager
    def context():
        with contextlib.ExitStack() as stack:
            if enable_loss_parallel:
                stack.enter_context(torch.distributed.tensor.parallel.loss_parallel())
            if enable_compiled_autograd:
                stack.enter_context(torch._dynamo.utils.maybe_enable_compiled_autograd(True))
            if enable_mixed_precision:
                stack.enter_context(
                    torch.autocast(device_type="cuda", dtype=TORCH_DTYPE_MAP[mixed_precision_param_dtype])
                )
            yield

    return context


def get_eval_context(enable_mixed_precision: bool, mixed_precision_param_dtype: str):
    @contextlib.contextmanager
    def context():
        with contextlib.ExitStack() as stack:
            stack.enter_context(torch.no_grad())
            if enable_mixed_precision:
                stack.enter_context(
                    torch.autocast(device_type="cuda", dtype=TORCH_DTYPE_MAP[mixed_precision_param_dtype])
                )
            yield

    return context


@record
def train_entry(job_config: TrainJobConfig):
    ao_flags = auto_patch(job_config.auto_patch.use_transformer_engine)
    gc_handler = GarbageCollection()
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    logger.info(f"Local rank: {local_rank}, world size: {world_size}")
    parallel_dims = ParallelDims(
        dp_replicate=job_config.training.dp_replicate,
        dp_shard=job_config.training.dp_shard,
        tp=job_config.training.tp_degree,
        pp=job_config.training.pp_degree,
        world_size=world_size,
        enable_loss_parallel=job_config.training.enable_loss_parallel,
        dp_type=job_config.training.data_parallel_type,
    )
    color = Color()
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    torch.cuda.set_device(device)
    init_distributed(dump_folder=job_config.training.dump_folder)
    world_mesh = parallel_dims.build_mesh(device_type="cuda")
    gpu_memory_monitor = build_gpu_memory_monitor()
    gpu_peak_flops = get_peak_flops(gpu_memory_monitor.device_name)
    # build callbacks first
    callbacks = build_callbacks(job_config.callbacks)
    if parallel_dims.dp_enabled:
        dp_mesh = world_mesh["dp"]
        dp_degree, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
    else:
        dp_degree, dp_rank = 1, 0
    # build model
    with torch.device("meta"), set_default_dtype(TORCH_DTYPE_MAP[job_config.model.torch_dtype]):
        model = build_model(job_config.model, ao_flags)
    # todo(xiaozhe): handle fp8 here
    logger.info(model)

    # model stats
    model_param_count = get_num_params(model)
    num_flop_per_token = get_num_flop_per_token(
        get_num_params(model, exclude_embedding=True),
        job_config.model,
        job_config.train_dataset.seq_len,
    )
    logger.info(f"Model has {humanize.intword(model_param_count)} parameters")
    # todo(xiaozhe): pipeline parallelism enabled
    # todo(xiaozhe): apply different parallelize function based on configurations
    parallelize_model(job_config.model.architecture, model, world_mesh, parallel_dims, train_config=job_config.training)
    init_device = "cpu" if job_config.checkpoint.create_seed_checkpoint else "cuda"
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
        f"{gpu_mem_stats.max_reserved_gib:.2f}GiB "
        f"({gpu_mem_stats.max_reserved_pct:.2f}%)"
    )
    # Build optimizer and scheduler
    optimizer = build_optimizer(model_parts, job_config.optimizer)
    scheduler = build_lr_scheduler(optimizer.optimizers, job_config)
    tokenizer = build_tokenizer(job_config.tokenizer.tokenizer_type, job_config.tokenizer.tokenizer_name_or_path)
    # build dataloader
    train_data_loader = build_hf_data_loader(
        job_config.train_dataset.name,
        job_config.train_dataset.path,
        job_config.train_dataset.stream,
        tokenizer,
        job_config.train_dataset.batch_size,
        job_config.train_dataset.seq_len,
        dp_degree,
        dp_rank,
    )
    if job_config.val_dataset is not None:
        val_data_loader = build_hf_data_loader(
            job_config.val_dataset.name,
            job_config.val_dataset.path,
            job_config.val_dataset.stream,
            tokenizer,
            job_config.val_dataset.batch_size,
            job_config.val_dataset.seq_len,
            dp_degree,
            dp_rank,
        )
    train_state = TrainState()

    # load initial checkpoint
    checkpoint = CheckpointManager(
        dataloader=train_data_loader,
        model_parts=model_parts,
        optimizers=optimizer.optimizers,
        lr_schedulers=scheduler.schedulers,
        states={"train_state": train_state},
        ckpt_config=job_config.checkpoint,
    )
    if job_config.checkpoint.create_seed_checkpoint:
        assert world_size == 1, "Must create seed-checkpoint using one gpu, to disable sharding"
        logger.info("Creating seed checkpoint")
        checkpoint.save(curr_step=0, force=True)
        logger.info("Seed checkpoint created, please restart training.")
        return
    logger.info(f"current device: {torch.cuda.current_device()}")
    checkpoint_loaded = checkpoint.load()
    if not checkpoint_loaded:
        raise ValueError("Failed to load checkpoint, please check.")
    metric_logger = build_metric_logger(job_config, parallel_dims)

    if train_state.step > 0:
        for idx, step in enumerate(train_state.log_steps):
            metrics = {
                "loss/global_avg_loss": train_state.global_avg_losses[idx],
                "loss/global_max_loss": train_state.global_max_losses[idx],
            }
            metric_logger.log(metrics, step=step)

    data_iterator = iter(train_data_loader)
    if job_config.val_dataset is not None:
        val_data_iterator = iter(val_data_loader)
    train_context = get_train_context(
        parallel_dims.loss_parallel_enabled,
        job_config.experimental.enable_compiled_autograd,
        enable_mixed_precision=True,
        mixed_precision_param_dtype=job_config.training.mixed_precision_param,
    )
    eval_context = get_eval_context(
        enable_mixed_precision=True,
        mixed_precision_param_dtype=job_config.training.mixed_precision_param,
    )
    losses_since_last_log = []
    ntokens_since_last_log = 0
    data_loading_times = []
    time_last_log = time.perf_counter()
    gpu_memory_monitor.reset_peak_stats()

    checkpoint.reset()
    logger.info(
        f"Training starts at step {train_state.step + 1}, "
        f"with {world_size} GPUs, "
        f"total consumed tokens {train_state.total_tokens}, "
        f"with local batch size {job_config.train_dataset.batch_size}, "
        f"global batch size {job_config.train_dataset.batch_size * dp_degree}, "
        f"sequence length {job_config.train_dataset.seq_len}, "
        f"total steps {job_config.training.train_steps} "
        f"(warmup {job_config.training.warmup_steps})"
    )
    with (
        maybe_enable_profiling(job_config, global_step=train_state.step) as torch_profiler,
        maybe_enable_memory_snapshot(job_config, global_step=train_state.step) as memory_profiler,
    ):
        while train_state.step < job_config.training.train_steps:
            train_state.step += 1
            gc_handler.run(train_state.step)
            # get train batch
            data_load_start = time.perf_counter()
            losses = 0
            optimizer.zero_grad()
            for microbatch_idx in range(job_config.training.accumulate_steps):
                batch = next(data_iterator)
                input_ids, labels = batch
                ntokens_since_last_log += labels.numel() * parallel_dims.dp_degree
                train_state.total_tokens += labels.numel() * parallel_dims.dp_degree
                data_loading_times.append(time.perf_counter() - data_load_start)
                input_ids = input_ids.cuda()
                labels = labels.cuda()
                with train_context():
                    pred = model(input_ids)
                    loss = cross_entropy_loss(pred, labels)
                    losses += loss
                    del pred

            losses.backward()

            for m in model_parts:
                torch.nn.utils.clip_grad_norm_(m.parameters(), job_config.training.max_norm, foreach=True)
            checkpoint.maybe_wait_for_staging()
            optimizer.step()
            scheduler.step()

            losses_since_last_log.append(loss)

            # log metrics
            if train_state.step == 1 or train_state.step % job_config.metrics.log_freq == 0:
                # forward pass on validation set
                val_losses = -1
                if job_config.val_dataset is not None:
                    model.eval()
                    with eval_context():
                        val_losses = 0
                        for _ in range(job_config.val_dataset.batch_size):
                            batch = next(val_data_iterator)
                            input_ids, labels = batch
                            input_ids = input_ids.cuda()
                            labels = labels.cuda()
                            pred = model(input_ids)
                            val_losses += cross_entropy_loss(pred, labels)
                            del pred
                        val_losses /= job_config.val_dataset.batch_size
                    model.train()

                losses = [loss.item() for loss in losses_since_last_log]
                avg_loss, max_loss = sum(losses) / len(losses), max(losses)
                if parallel_dims.dp_enabled:
                    global_avg_loss, global_max_loss = (
                        dist_mean(avg_loss, dp_mesh),
                        dist_max(max_loss, dp_mesh),
                    )
                else:
                    global_avg_loss, global_max_loss = avg_loss, max_loss

                # update train state
                train_state.log_steps.append(train_state.step)
                train_state.global_avg_losses.append(global_avg_loss)
                train_state.global_max_losses.append(global_max_loss)

                time_delta = time.perf_counter() - time_last_log

                # tokens per second, abbr. as tps
                tps = ntokens_since_last_log / time_delta
                # model FLOPS utilization
                # For its definition and calculation, please refer to the PaLM paper:
                # https://arxiv.org/abs/2204.02311
                mfu = 100 * num_flop_per_token * tps / gpu_peak_flops / parallel_dims.world_size

                tpd = ntokens_since_last_log / time_delta / parallel_dims.world_size

                time_end_to_end = time_delta / job_config.metrics.log_freq
                time_data_loading = sum(data_loading_times) / len(data_loading_times)
                time_data_loading_pct = 100 * sum(data_loading_times) / time_delta

                gpu_mem_stats = gpu_memory_monitor.get_peak_stats()

                metrics = {
                    "Total Tokens": train_state.total_tokens,
                    "loss/global_avg_loss": global_avg_loss,
                    "loss/global_max_loss": global_max_loss,
                    "loss/val_loss": val_losses,
                    "perf/Tokens Per Second": tps,
                    "perf/MFU (%)": mfu,
                    "perf/Tokens Per Second Per GPU": tpd,
                    "config/learning_rate": optimizer.learning_rate,
                    "time/end_to_end (s)": time_end_to_end,
                    "time/data_loading (s)": time_data_loading,
                    "time/data_loading (%)": time_data_loading_pct,
                    "memory/max_active (GiB)": gpu_mem_stats.max_active_gib,
                    "memory/max_active (%)": gpu_mem_stats.max_active_pct,
                    "memory/max_reserved (GiB)": gpu_mem_stats.max_reserved_gib,
                    "memory/max_reserved (%)": gpu_mem_stats.max_reserved_pct,
                    "memory/num_alloc_retries": gpu_mem_stats.num_alloc_retries,
                    "memory/num_ooms": gpu_mem_stats.num_ooms,
                }
                metric_logger.log(metrics, step=train_state.step)
                logger.info(
                    f"{color.cyan}step: {train_state.step:2}  "
                    f"{color.green}loss: {global_avg_loss:7.4f}  "
                    f"{color.green}val_loss: {val_losses:7.4f}  "
                    f"{color.yellow}memory: {gpu_mem_stats.max_reserved_gib:5.2f}GiB"
                    f"({gpu_mem_stats.max_reserved_pct:.2f}%)  "
                    f"{color.blue}tps: {round(tps):,}  "
                    f"{color.magenta}mfu: {mfu:.2f}%  "
                    f"{color.red}tokens: {humanize.intword(train_state.total_tokens)}{color.reset}"
                )

                losses_since_last_log.clear()
                ntokens_since_last_log = 0
                data_loading_times.clear()
                time_last_log = time.perf_counter()
                gpu_memory_monitor.reset_peak_stats()

            for callback in callbacks:
                callback.step(
                    train_state.step,
                    {
                        "model": model,
                        "tokenizer": tokenizer,
                        "metric_logger": metric_logger,
                    },
                )

            # explicitly update train state
            # TODO(xiaozhe): we should pass a pointer instead of a copy, but I don't know why it doesn't work
            checkpoint.update_states(
                train_state.step,
                force=(train_state.step == job_config.training.train_steps),
                train_state=train_state,
                dataloader=train_data_loader,
                model_parts=model_parts,
                optimizers=optimizer.optimizers,
                lr_schedulers=scheduler.schedulers,
            )

            checkpoint.save(train_state.step, force=(train_state.step == job_config.training.train_steps))

            # signal the profiler that the next profiling step has started
            if torch_profiler:
                torch_profiler.step()
            if memory_profiler:
                memory_profiler.step()

                # reduce timeout after first train step for faster signal
                # (assuming lazy init and compilation are finished)
                if train_state.step == 1:
                    set_pg_timeouts(
                        timeout=timedelta(seconds=job_config.comm.train_timeout_seconds),
                        world_mesh=world_mesh,
                    )

        if torch.distributed.get_rank() == 0:
            logger.info("Sleeping 2 seconds for other ranks to complete")
            time.sleep(2)

        metric_logger.close()
        logger.info("Training completed")

    torch.distributed.destroy_process_group()
