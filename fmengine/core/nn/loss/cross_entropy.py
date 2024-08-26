import torch

def cross_entropy_loss(pred, labels):
    return torch.nn.functional.cross_entropy(pred.flatten(0, 1), labels.flatten(0, 1))
