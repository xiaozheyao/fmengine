import os
import torch
import transformers
from omegaconf import OmegaConf
from torch.fx import GraphModule
from typing import Optional

from fmengine.core.parallelism.distributed import init_distributed
from fmengine.core.configs import TORCH_DTYPE_MAP
from fmengine.core.configs.train_config import TrainJobConfig
from fmengine.cli.utils import enforce_nondistributed_env
from fmengine.models.builder import import_from_huggingface
from fmengine.utilities import logger, auto_patch, set_default_dtype
from fmengine.models.builder import build_model
from fmengine.core.checkpoint import CheckpointManager, TrainState
from fmengine.core.nn import build_lr_scheduler, build_optimizer
from fmengine.data import build_hf_data_loader
from fmengine.data.tokenizer import build_tokenizer


def inference_entry(model_id: str, revision: Optional[str], prompt: str, temperature: float, top_k: int, top_p: float):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_id, revision=revision).to("cuda")
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    generation_output = model.generate(**inputs, return_dict_in_generate=True, output_scores=True, max_new_tokens=128)

    print(f"scores (first token): {generation_output.scores[0]}")
    output_texts = tokenizer.batch_decode(generation_output.sequences, skip_special_tokens=True)
    print(f"Generated text: {output_texts}")
    return generation_output


def prepare_ckpt_entry(job_config: TrainJobConfig, config_file: str):
    ao_flags = auto_patch(job_config.auto_patch.use_transformer_engine)
    enforce_nondistributed_env()
    train_state = TrainState()
    initialization_required = True
    # get dir of the config file
    config_dir = os.path.dirname(config_file)
    init_distributed(dump_folder=job_config.training.dump_folder)
    os.makedirs(job_config.checkpoint.ckpt_dir, exist_ok=True)
    torch.cuda.set_device(f"cuda:0")

    if job_config.checkpoint.finetuned_from is not None:
        initialization_required = False
        logger.info(f"Converting pretrained model from {job_config.checkpoint.finetuned_from}")
        logger.warning(
            f"The model config from the checkpoint will be used instead of the config from the job config. New configuration will be saved under {config_dir}/model_def.yaml"
        )
        model, config = import_from_huggingface(
            job_config.model.architecture,
            job_config.checkpoint.finetuned_from,
            job_config.checkpoint.export_dtype,
            ao_flags,
        )
        with open(f"{config_dir}/model_def.yaml", "w+") as f:
            OmegaConf.save({"model": config}, f)
    else:
        logger.info(f"Building model from scratch")
        with torch.device("meta"), set_default_dtype(TORCH_DTYPE_MAP[job_config.model.torch_dtype]):
            model = build_model(job_config.model, ao_flags)
        model.to_empty(device="cpu")

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
        job_config.train_dataset.name,
        job_config.train_dataset.path,
        job_config.train_dataset.stream,
        tokenizer,
        job_config.train_dataset.batch_size,
        job_config.train_dataset.seq_len,
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
