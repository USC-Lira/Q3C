import torch as th

## SMOOTHING DECAY FUNCTIONS ##

class ExponentialSmoothingDecay:
    def __init__(self, initial_value: float, min_value: float, device: th.device):
        self.initial_value = th.tensor([initial_value]).to(device)
        self.min_value = min_value

    def __call__(self, num_timesteps, total_timesteps):
        """
        Applies exponential decay to a given C value.

        :param num_timesteps: Current number of timesteps.
        :param total_timesteps: Total number of timesteps for training.
        :return: The decayed C value as a torch.Tensor.
        """
        decay_factor = num_timesteps / total_timesteps
        decayed_C = self.min_value + (self.initial_value - self.min_value) * (1 - decay_factor) ** 2
        return decayed_C


class DelayedExponentialSmoothingDecay: 
    def __init__(self, initial_value: float, min_value: float, device: th.device): 
        self.initial_value = th.tensor([initial_value]).to(device)
        self.min_value = min_value

    def __call__(self, num_timesteps, total_timesteps): 
        """
        Applies inverse exponential decay to a given C value.

        :param num_timesteps: Current number of timesteps.
        :param total_timesteps: Total number of timesteps for training.
        :return: The decayed C value as a torch.Tensor.
        """
        progress_remaining = ((total_timesteps - num_timesteps) / total_timesteps) ** 0.2
        return self.initial_value * progress_remaining + self.min_value * (1 - progress_remaining)
 
    
    
class DelayedExponentialSmoothingDecay: 
    def __init__(self, initial_value: float, min_value: float, device: th.device): 
        self.initial_value = th.tensor([initial_value]).to(device)
        self.min_value = min_value

    def __call__(self, num_timesteps, total_timesteps): 
        """
        Applies inverse exponential decay to a given C value.

        :param num_timesteps: Current number of timesteps.
        :param total_timesteps: Total number of timesteps for training.
        :return: The decayed C value as a torch.Tensor.
        """
        progress_remaining = ((total_timesteps - num_timesteps) / total_timesteps) ** 0.2
        return self.initial_value * progress_remaining + self.min_value * (1 - progress_remaining)
 
    
class LinearSmoothingDecay:
    def __init__(self, initial_value: float, min_value: float, device: th.device):
        self.initial_value = th.tensor([initial_value]).to(device)
        self.min_value = min_value

    def __call__(self, num_timesteps, total_timesteps):
        """
        Applies linear decay to a given C value.

        :param num_timesteps: Current number of timesteps.
        :param total_timesteps: Total number of timesteps for training.
        :return: The decayed C value as a torch.Tensor.
        """
        decay_factor = num_timesteps / total_timesteps
        decayed_C = self.min_value + (self.initial_value - self.min_value) * (1 - decay_factor)
        return decayed_C
