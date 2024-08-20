from typing import Optional

from fmengine.datasets.dataloader.dp_dataloader import DPAwareDataLoader
from fmengine.datasets.huggingface import HuggingFaceDataset
from fmengine.datasets.tokenizer import Tokenizer


def build_hf_data_loader(
    dataset_name: str,
    dataset_path: Optional[str],
    streaming: bool,
    tokenizer: Tokenizer,
    batch_size: int,
    seq_len: int,
    world_size,
    rank,
    infinite: bool = True,
):
    hf_ds = HuggingFaceDataset(dataset_name, dataset_path, tokenizer, seq_len, world_size, rank, infinite, streaming)

    return DPAwareDataLoader(rank, hf_ds, batch_size=batch_size)
