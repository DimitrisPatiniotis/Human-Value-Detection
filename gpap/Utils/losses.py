import torch

def F1_as_evaluator(y_pred, y_true, target_cols):
    """
    F1 computed as in https://github.com/touche-webis-de/touche-code/blob/main/semeval23/human-value-detection/evaluator/evaluator.py

    :param y_pred: numpy array with the predicted values. Shape: [total_batches * batch_size, length of target_cols]
    :param y_true: numpy array of the true values. Shape: [total_batches * batch_size, length of target_cols]
    :param target_cols: List of the selected values
    :return: fmeasure, precision, recall, fmeasures, precisions, recalls
    """

    def initializeCounter(availableValues):
        counter = {}
        for value_ in availableValues:
            counter[value_] = 0
        return counter

    relevants = initializeCounter(target_cols)
    positives = initializeCounter(target_cols)
    truePositives = initializeCounter(target_cols)

    for labels in y_true:
        for value, label in enumerate(labels):
            if label == 1:
                relevants[target_cols[value]] += 1

    for argumentId, labels in enumerate(y_pred):
        for value, label in enumerate(labels):
            if label == 1:
                positives[target_cols[value]] += 1
                if y_true[argumentId][value] == 1:
                    truePositives[target_cols[value]] += 1

    precisions = []
    recalls = []
    fmeasures = []
    for value in target_cols:
        if relevants[value] != 0:
            precision = 0
            if positives[value] != 0:
                precision = truePositives[value] / positives[value]
            precisions.append(precision)
            recall = truePositives[value] / relevants[value]
            recalls.append(recall)
            fmeasure = 0
            if precision + recall != 0:
                fmeasure = 2 * precision * recall / (precision + recall)
            fmeasures.append(fmeasure)
    precision = sum(precisions) / len(precisions)
    recall = sum(recalls) / len(recalls)

    fmeasure = 0
    if precision + recall != 0:
        fmeasure = 2 * precision * recall / (precision + recall)

    return fmeasure, precision, recall, fmeasures, precisions, recalls

def BCE_loss(outputs, targets, weights=None):

    return torch.nn.BCEWithLogitsLoss(weight=weights)(outputs, targets)

def f1_loss(y_pred, y_true, weights=None):
    """
    Calculate F1 score. Can work with gpu tensors.

    Reference
    ---------
    - https://gist.github.com/SuperShinyEyes/dcc68a08ff8b615442e3bc6a9b55a354
    - Paper: sigmoidF1: A Smooth F1 Score Surrogate Loss for Multilabel Classification

    """

    sig_ = torch.nn.Sigmoid()
    y_pred_probs = sig_(y_pred)

    tp = (y_true * y_pred_probs).to(torch.float32)
    fp = ((1 - y_true) * y_pred_probs).to(torch.float32)
    fn = (y_true * (1 - y_pred_probs)).to(torch.float32)

    if weights is not None:
        tp = tp * weights
        fp = fp * weights
        fn = fn * weights

    tp = tp.mean(dim=0)
    fp = fp.mean(dim=0)
    fn = fn.mean(dim=0)

    epsilon = 1e-7

    sigmoidF1 = (2*tp) / ((2*tp) + fn + fp + epsilon)
    mean_sigmoidF1 = sigmoidF1.mean()

    reverse_mean_sigmoidF1 = 1.00 - mean_sigmoidF1

    return reverse_mean_sigmoidF1