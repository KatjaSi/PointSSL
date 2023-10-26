import torch
import torch.nn.functional as F

def cross_entropy_loss_with_label_smoothing(pred, gold, smoothing=True):
    '''
    Calculate cross entropy loss, optionally apply label smoothing.

    Parameters:
        pred (torch.Tensor): The predicted logits from the model.
                            Shape: (batch_size, num_classes).
        gold (torch.Tensor): The true class labels.
                            Shape: (batch_size,).
        smoothing (bool, optional): Flag indicating whether to apply label smoothing.
                                   Default is True.

    Notes:
        If `smoothing` is set to True, label smoothing will be applied. Label smoothing
        helps prevent the model from becoming overconfident by adding a small amount of
        uncertainty to the true labels.
    '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss