import os
import torch
import transformers
from omegaconf import OmegaConf
from torch.fx import GraphModule
from fmengine.core.parallelism.distributed import init_distributed
from fmengine.core.configs.train_config import TrainJobConfig
from fmengine.cli.utils import enforce_nondistributed_env
from fmengine.models.builder import import_from_huggingface
from fmengine.utilities import logger
from fmengine.models.builder import build_model
from fmengine.core.checkpoint import CheckpointManager, TrainState
from fmengine.core.nn import build_lr_scheduler, build_optimizer
from fmengine.datasets import build_hf_data_loader
from fmengine.datasets.tokenizer import build_tokenizer

def inference_entry(model_id: str, prompt: str, temperature: float, top_k: int, top_p: float):
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        do_sample=True,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device="cuda",
        max_new_tokens=128,
    )
    output = pipeline(
        prompt,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    return output


def prepare_ckpt_entry(job_config: TrainJobConfig, config_file: str):
    enforce_nondistributed_env()
    train_state = TrainState()
    initialization_required = True
    # get dir of the config file
    config_dir = os.path.dirname(config_file)
    init_distributed(dump_folder=job_config.training.dump_folder)
    os.makedirs(job_config.checkpoint.ckpt_dir, exist_ok=True)
    if job_config.checkpoint.finetuned_from is not None:
        initialization_required = False
        logger.info(f"Converting pretrained model from {job_config.checkpoint.finetuned_from}")
        logger.warning(f"The model config from the checkpoint will be used instead of the config from the job config. New configuration will be saved under {job_config.checkpoint.ckpt_dir}/model.yaml")

        model, config = import_from_huggingface(job_config.model.architecture, job_config.checkpoint.finetuned_from, job_config.checkpoint.export_dtype)

        with open(f"{config_dir}/model_def.yaml", "w+") as f:
            OmegaConf.save({'model': config}, f)
    else:
        logger.info(f"Building model from scratch")
        with torch.device("meta"):
            model = build_model(job_config.model)
    model_parts = [model]

    for mod in model_parts:
        # skip traced modules since we do not define init_weights in the traced module
        if isinstance(mod, GraphModule):
            continue
        if initialization_required:
            mod.init_weights()
        mod.train()
    optimizer = build_optimizer(model_parts, job_config.optimizer)
    scheduler = build_lr_scheduler(optimizer.optimizers, job_config)
    tokenizer = build_tokenizer(job_config.tokenizer.tokenizer_type, job_config.tokenizer.tokenizer_name_or_path)
    # build dataloader
    data_loader = build_hf_data_loader(
        job_config.dataset.name,
        job_config.dataset.path,
        job_config.dataset.stream,
        tokenizer,
        job_config.dataset.batch_size,
        job_config.dataset.seq_len,
        1,
        1,
    )
    checkpoint = CheckpointManager(
        dataloader=data_loader,
        model_parts=model_parts,
        optimizers=optimizer.optimizers,
        lr_schedulers=scheduler.schedulers,
        states={"train_state": train_state},
        ckpt_config=job_config.checkpoint,
    )
    checkpoint.save(curr_step=0, force=True)
    logger.info("Seed checkpoint created, please start training.")
    torch.distributed.destroy_process_group()
    return
