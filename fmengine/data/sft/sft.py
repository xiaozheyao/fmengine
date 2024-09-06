from typing import List, Optional
import torch
from datasets import Dataset, load_dataset
from datasets.distributed import split_dataset_by_node
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from fmengine.utilities import logger

CROSS_ENTROPY_IGNORE_IDX = -100


class SFTDataset(IterableDataset, Stateful):
    pass
