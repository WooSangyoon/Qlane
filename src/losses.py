import torch


def masked_mse_loss(pred, target, mask, eps=1e-6):
    """
    Masked Mean Squared Error loss.

    Args:
        pred:   Tensor, shape (B, K, H)
        target: Tensor, shape (B, K, H)
        mask:   Tensor, shape (B, K, H), values in {0, 1}
        eps:    Small value to avoid division by zero

    Returns:
        Scalar tensor (loss)
    """
    # ensure same shape
    assert pred.shape == target.shape == mask.shape

    # squared error
    diff = pred - target
    diff2 = diff * diff

    # apply mask
    diff2 = diff2 * mask

    # normalize by number of valid entries
    valid_count = mask.sum()

    loss = diff2.sum() / (valid_count + eps)
    return loss


def masked_l1_loss(pred, target, mask, eps=1e-6):
    """
    Masked Mean Absolute Error loss.
    """
    assert pred.shape == target.shape == mask.shape

    diff = torch.abs(pred - target)
    diff = diff * mask

    valid_count = mask.sum()

    loss = diff.sum() / (valid_count + eps)
    return loss
