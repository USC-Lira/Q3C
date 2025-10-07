import torch

def compute_penalty(mu, sigma=0.5, lambda_penalty=1.0):
    """
    Compute the penalty loss for multiple actors' actions in a vectorized manner.

    Args:
        mu (torch.Tensor): Tensor of shape [batch_size, N, action_dim],
                           where N is the number of actors.
        sigma (float, optional): Controls the width of the penalty region.
                                  Defaults to 0.5.
        lambda_penalty (float, optional): Weight of the penalty term.
                                         Defaults to 1.0.

    Returns:
        torch.Tensor: A single scalar tensor representing the penalty loss.
    """
    batch_size, num_actors, action_dim = mu.shape

    if num_actors < 2:
        # No penalty needed if there's only one actor
        return torch.tensor(0.0, device=mu.device)

    # Detach previous_mu to prevent gradient flow into previous actors
    mu_detached = mu.detach()  # Shape: [batch_size, N, action_dim]

    # Compute pairwise differences between actors
    # Shape: [batch_size, N, N, action_dim]
    diff = mu.unsqueeze(2) - mu_detached.unsqueeze(1)

    # Compute squared Euclidean distances
    # Shape: [batch_size, N, N]
    squared_distance = (diff ** 2).sum(dim=-1)

    # Create a mask to consider only j < i (lower triangular without diagonal)
    # Shape: [N, N]
    mask = torch.tril(torch.ones((num_actors, num_actors), device=mu.device), diagonal=-1)

    # Expand mask to match batch size
    # Shape: [batch_size, N, N]
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)

    # Apply the mask to the squared distances
    # Only distances where j < i are considered
    masked_squared_distance = squared_distance * mask

    # Compute Gaussian penalties
    # Shape: [batch_size, N, N]
    penalties = torch.exp(-masked_squared_distance / (2 * sigma ** 2)) * mask

    # Sum all penalties across actors and mean across batches
    total_penalty = penalties.sum()

    # Compute the average penalty loss
    penalty_loss = (lambda_penalty * total_penalty) / batch_size

    return penalty_loss
