import pickle
from typing import Any, Dict, List, Optional

import torch
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset
