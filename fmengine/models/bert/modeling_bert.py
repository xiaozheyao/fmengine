from transformers import BertPreTrainedModel, BertModel
from .config_bert import BertArgs
from torch import nn
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from fmengine.core.configs.train_config import AutoOptimizationFlags


def build_bert(args: BertArgs, ao_flags: Optional["AutoOptimizationFlags"]) -> nn.Module:
    pass
