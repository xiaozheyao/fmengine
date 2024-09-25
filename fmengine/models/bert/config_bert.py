from dataclasses import dataclass
from typing import Optional
from transformers import BertConfig


@dataclass
class BertArgs:
    architecture: str = "bert"
    attention_probs_dropout_prob: float = 0.1
    bos_token_id: int
    eos_token_id: int
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    hidden_size: int = 768
    initializer_range: float = 0.02
    intermediate_size: int = 3072
    layer_norm_eps: float = 1e-05
    max_position_embeddings: int = 514
    num_attention_heads: int = 12
    num_hidden_layers: int = 12
    pad_token_id: int
    type_vocab_size: int = 2
    vocab_size: int = 50265

    @classmethod
    def from_pretrained(cls, pretrained_model_id_or_path: str):
        config = BertConfig.from_pretrained(pretrained_model_id_or_path)
        return cls(
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            bos_token_id=config.bos_token_id,
            eos_token_id=config.eos_token_id,
            hidden_act=config.hidden_act,
            hidden_dropout_prob=config.hidden_dropout_prob,
            hidden_size=config.hidden_size,
            initializer_range=config.initializer_range,
            intermediate_size=config.intermediate_size,
            layer_norm_eps=config.layer_norm_eps,
            max_position_embeddings=config.max_position_embeddings,
            num_attention_heads=config.num_attention_heads,
            num_hidden_layers=config.num_hidden_layers,
            pad_token_id=config.pad_token_id,
            type_vocab_size=config.type_vocab_size,
            vocab_size=config.vocab_size,
        )
