import torch
from typing import Dict, Any
from .config_llama import LlamaArgs
from transformers import LlamaConfig, AutoModelForCausalLM
from fmengine.models.builder import build_model


def to_huggingface(states: Dict[str, Any], export_dtype: str, config: LlamaArgs):
    # step 1: create a config file containing the model architecture
    config = LlamaConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_dim,
        num_hidden_layers=config.n_layers,
        num_attention_heads=config.n_heads,
        num_key_value_heads=config.n_kv_heads,
        hidden_act=config.activation,
        max_position_embeddings=config.max_seq_len,
        initializer_range=0.02,
        rope_theta=config.rope_theta,
    )
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    # step 2: load the model weights, we need some translation here
    new_state_dict = {}
    with torch.no_grad():
        ## step 2.1: handle non-transformer-blocks
        new_state_dict["model.embed_tokens.weight"] = states["model"]["tok_embeddings.weight"]
        new_state_dict["model.norm.weight"] = states["model"]["norm.weight"]
        new_state_dict["lm_head.weight"] = states["model"]["output.weight"]
        ## step 2.2: handle transformer blocks
        for i in range(config.num_hidden_layers):
            new_state_dict[f"model.layers.{i}.self_attn.q_proj.weight"] = states["model"][
                f"layers.{i}.attn.q_proj.weight"
            ]
            new_state_dict[f"model.layers.{i}.self_attn.k_proj.weight"] = states["model"][
                f"layers.{i}.attn.k_proj.weight"
            ]
            new_state_dict[f"model.layers.{i}.self_attn.v_proj.weight"] = states["model"][
                f"layers.{i}.attn.v_proj.weight"
            ]
            new_state_dict[f"model.layers.{i}.self_attn.o_proj.weight"] = states["model"][
                f"layers.{i}.attn.output_proj.weight"
            ]
            new_state_dict[f"model.layers.{i}.input_layernorm.weight"] = states["model"][
                f"layers.{i}.self_attn_norm.weight"
            ]
            new_state_dict[f"model.layers.{i}.post_attention_layernorm.weight"] = states["model"][
                f"layers.{i}.mlp_norm.weight"
            ]
            new_state_dict[f"model.layers.{i}.mlp.gate_proj.weight"] = states["model"][f"layers.{i}.mlp.w1.weight"]
            new_state_dict[f"model.layers.{i}.mlp.down_proj.weight"] = states["model"][f"layers.{i}.mlp.w2.weight"]
            new_state_dict[f"model.layers.{i}.mlp.up_proj.weight"] = states["model"][f"layers.{i}.mlp.w3.weight"]

        from fmengine.core.configs import TORCH_DTYPE_MAP

        export_dtype = TORCH_DTYPE_MAP[export_dtype]
        new_state_dict = {k: v.to(export_dtype) for k, v in new_state_dict.items()}
        model.load_state_dict(new_state_dict, strict=True, assign=True)
    model.eval()
    return model, config


def from_huggingface(pretrained_model_id_or_path: str, load_dtype: str):
    from fmengine.core.configs import TORCH_DTYPE_MAP

    model = AutoModelForCausalLM.from_pretrained(pretrained_model_id_or_path, torch_dtype=TORCH_DTYPE_MAP[load_dtype])
    state_dict = model.state_dict()
    config = LlamaConfig.from_pretrained(pretrained_model_id_or_path)
    fmengine_config = LlamaArgs(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        intermediate_dim=config.intermediate_size,
        n_layers=config.num_hidden_layers,
        n_heads=config.num_attention_heads,
        n_kv_heads=config.num_key_value_heads,
        activation=config.hidden_act,
        max_seq_len=config.max_position_embeddings,
        rope_theta=config.rope_theta,
    )
    with torch.device("meta"):
        fmengine_model = build_model(fmengine_config)
    model_state_dict = {}
    with torch.no_grad():
        model_state_dict["tok_embeddings.weight"] = state_dict["model.embed_tokens.weight"]
        model_state_dict["norm.weight"] = state_dict["model.norm.weight"]
        model_state_dict["output.weight"] = state_dict["lm_head.weight"]
        for i in range(config.num_hidden_layers):
            model_state_dict[f"layers.{i}.attn.q_proj.weight"] = state_dict[f"model.layers.{i}.self_attn.q_proj.weight"]
            model_state_dict[f"layers.{i}.attn.k_proj.weight"] = state_dict[f"model.layers.{i}.self_attn.k_proj.weight"]
            model_state_dict[f"layers.{i}.attn.v_proj.weight"] = state_dict[f"model.layers.{i}.self_attn.v_proj.weight"]
            model_state_dict[f"layers.{i}.attn.output_proj.weight"] = state_dict[
                f"model.layers.{i}.self_attn.o_proj.weight"
            ]
            model_state_dict[f"layers.{i}.self_attn_norm.weight"] = state_dict[
                f"model.layers.{i}.input_layernorm.weight"
            ]
            model_state_dict[f"layers.{i}.mlp_norm.weight"] = state_dict[
                f"model.layers.{i}.post_attention_layernorm.weight"
            ]
            model_state_dict[f"layers.{i}.mlp.w1.weight"] = state_dict[f"model.layers.{i}.mlp.gate_proj.weight"]
            model_state_dict[f"layers.{i}.mlp.w2.weight"] = state_dict[f"model.layers.{i}.mlp.down_proj.weight"]
            model_state_dict[f"layers.{i}.mlp.w3.weight"] = state_dict[f"model.layers.{i}.mlp.up_proj.weight"]

    fmengine_model.load_state_dict(model_state_dict, strict=True, assign=True)
    return fmengine_model, fmengine_config
