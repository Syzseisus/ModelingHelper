import torch


def accuracy(logit, label):
    _, preds = torch.max(logit, 1)
    correct = (preds == label).sum().item()

    return 100 * correct / len(label)
