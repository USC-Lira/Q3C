import torch as th

## ENTROPY REGULATOR LOSS FUNCTIONS ##

def min_n_pairwise_distances_loss(points):
    B, N, D = points.shape  # Batch size, number of points, number of dimensions
    pairwise_dists = th.cdist(points, points, p=2)  # Shape: [B, N, N]
    mask = th.eye(N, device=points.device).bool()
    pairwise_dists.masked_fill(mask, 0)
    return -pairwise_dists.flatten(1).sort(dim=1)[0][:, :2 * N].sum() / (B * N)


def repulsion_loss(points, epsilon=1e-6):
    """
    Computes a repulsion loss for a batch of point sets.

    Parameters:
        points (th.Tensor): Tensor of shape [B, N, D], where B is batch size,
                            N is number of points per batch, and D is the number of dimensions.
        epsilon (float): A small constant added to distances for numerical stability.

    Returns:
        th.Tensor: A scalar tensor representing the repulsion loss.
    """
    B, N, D = points.shape
    dists = th.cdist(points, points, p=2)
    mask = ~th.eye(N, device=points.device).bool()  # shape: [N, N]
    dists_off_diag = dists.masked_select(mask).view(B, N, N - 1)
    repulsion = 1.0 / (dists_off_diag + epsilon)
    loss_per_batch = repulsion.mean(dim=[1, 2])
    return loss_per_batch.mean()


def maximin_loss(points, beta=100.0, epsilon=1e-6):
    """
    Computes a maximin loss for a batch of point sets using a soft-min approximation.

    The loss is defined as the negative average (over points and batches) of the soft-minimum
    of the distances from each point to all other points. Minimizing this loss encourages
    the network to maximize the minimum pairwise distance.

    Parameters:
        points (th.Tensor): Tensor of shape [B, N, D], where B is the batch size, N is the number
                            of points per batch, and D is the dimensionality of each point.
        beta (float): Parameter controlling the sharpness of the soft-min approximation. A larger
                    beta makes the approximation closer to the hard minimum.
        epsilon (float): Small constant for numerical stability.

    Returns:
        th.Tensor: A scalar tensor representing the maximin loss.
    """
    B, N, D = points.shape
    dists = th.cdist(points, points, p=2)
    mask = ~th.eye(N, device=points.device).bool()
    dists_masked = dists.masked_select(mask).view(B, N, N - 1) + epsilon
    weights = th.softmax(-beta * dists_masked, dim=-1)
    soft_min = (dists_masked * weights).sum(dim=-1)  # Shape: [B, N]
    avg_soft_min = soft_min.mean()
    return -avg_soft_min

