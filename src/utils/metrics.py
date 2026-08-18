"""
Metrics utilities: Top-K accuracy, per-class precision/recall/F1,
confusion matrix, and McNemar's test helper.
"""
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def topk_accuracy(outputs, targets, topk=(1, 2, 3)):
    """Returns a dict {f'top{k}': float} for each k in topk.
    outputs: (N, num_classes) numpy array or torch tensor (logits or probs)
    targets: (N,) ground-truth class indices
    """
    import torch
    if not isinstance(outputs, torch.Tensor):
        outputs = torch.tensor(outputs)
    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets)
    # bfloat16 (XPU AMP) is not supported by topk on all backends — cast to float32
    outputs = outputs.float()

    maxk = max(topk)
    _, pred = outputs.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    result = {}
    for k in topk:
        correct_k = correct[:k].any(dim=0).float().sum().item()
        result[f"top{k}"] = correct_k / targets.size(0) * 100.0
    return result


def per_class_report(targets, preds, class_names):
    """Returns sklearn classification_report dict (precision/recall/F1 per class).
    Always reports all classes even if some don't appear in a small eval subset."""
    return classification_report(
        targets, preds,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )


def get_confusion_matrix(targets, preds, class_names):
    """Returns (cm, class_names) where cm is a numpy (C, C) array."""
    cm = confusion_matrix(targets, preds, labels=list(range(len(class_names))))
    return cm, class_names


def mcnemar_test(targets, preds_a, preds_b):
    """McNemar's test: are model A and B statistically different?
    Returns (statistic, p_value). Use scipy.
    targets, preds_a, preds_b: 1-D arrays of class indices.
    """
    from scipy.stats import chi2
    n01 = np.sum((preds_a == targets) & (preds_b != targets))  # A right, B wrong
    n10 = np.sum((preds_a != targets) & (preds_b == targets))  # A wrong, B right
    # With continuity correction
    if (n01 + n10) == 0:
        return 0.0, 1.0
    stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    p = 1 - chi2.cdf(stat, df=1)
    return stat, p
