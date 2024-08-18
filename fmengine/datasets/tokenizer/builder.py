from fmengine.utilities import logger

from .huggingface import HFTokenizer
from .tiktoken import TikTokenizer
from .tokenizer import Tokenizer


def build_tokenizer(tokenizer_type: str, tokenizer_path: str) -> Tokenizer:
    logger.info(f"Building {tokenizer_type} tokenizer locally from {tokenizer_path}")
    if tokenizer_type == "huggingface":
        return HFTokenizer(tokenizer_path)
    elif tokenizer_type == "tiktoken":
        return TikTokenizer(tokenizer_path)
    else:
        raise ValueError(f"Unknown tokenizer type: {args.type}")
