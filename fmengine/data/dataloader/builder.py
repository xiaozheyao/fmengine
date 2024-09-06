from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from fmengine.data.tokenizer import Tokenizer


def build_hf_data_loader(
    dataset_name: str,
    dataset_path: Optional[str],
    streaming: bool,
    tokenizer: "Tokenizer",
    batch_size: int,
    seq_len: int,
    world_size,
    rank,
    infinite: bool = True,
):
    from fmengine.data.dataloader.dp_dataloader import DPAwareDataLoader
    from fmengine.data.huggingface import HuggingFaceDataset

    hf_ds = HuggingFaceDataset(dataset_name, dataset_path, tokenizer, seq_len, world_size, rank, infinite, streaming)
    return DPAwareDataLoader(rank, hf_ds, batch_size=batch_size)
