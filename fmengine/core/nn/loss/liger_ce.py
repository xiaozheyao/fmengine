import torch
from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss

def liger_cross_entropy_loss(pred, labels):
    # TODO(xiaozhe): this doesn't seem to work for me
    return LigerCrossEntropyLoss(pred.flatten(0, 1), labels.flatten(0, 1))
