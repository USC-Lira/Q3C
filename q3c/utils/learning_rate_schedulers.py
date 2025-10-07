import math
from typing import Callable

def delayed_exponential_schedule(initial_learning_rate: float) -> Callable[[float], float]:
    """
    Delayed exponential decay learning rate schedule.


    :param initial_value: Initial learning rate.
    :return: schedule that computes
     current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
       """
       Progress will decrease from 1 (beginning) to 0.


       :param progress_remaining:
       :return: current learning rate
       """
       final_value = initial_learning_rate * 1e-1
       progress_remaining = progress_remaining ** 0.2 # slows down decay
       return initial_learning_rate * progress_remaining + final_value * (1 - progress_remaining)


    return func

def exponential_schedule(initial_learning_rate: float, total_timesteps: int, warmup_steps: int = 1_000_000) -> Callable[[float], float]:
    """
    Exponential learning rate schedule that decays to 10% over `warmup_steps`, then continues decaying exponentially beyond that.
    
    :param initial_learning_rate: Starting learning rate.
    :param warmup_steps: Step at which LR reaches 10% of initial.
    :return: Schedule function returning LR per progress remaining.
    """
    decay_rate = math.log(0.5) / warmup_steps  # Ensures 10% at warmup_steps
    
    def schedule(progress_remaining: float) -> float:
        current_step = int((1 - progress_remaining) * total_timesteps)
        return initial_learning_rate * math.exp(decay_rate * current_step)
    
    return schedule


def linear_schedule(initial_learning_rate: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule that decays from `initial_learning_rate` to 10% of it over `total_timesteps`.

    :param initial_learning_rate: Starting learning rate.
    :return: Schedule function returning LR per progress remaining.
    """
    final_learning_rate = initial_learning_rate * 0.5
    slope = (final_learning_rate - initial_learning_rate)

    def schedule(progress_remaining: float) -> float:
        return initial_learning_rate + slope * (1 - progress_remaining)

    return schedule

def one_cycle_lr_schedule(max_lr: float, pct_start: float = 0.3, div_factor: float = 2.0) -> Callable[[float], float]:
    """
    Implements a OneCycleLR-style learning rate schedule using cosine annealing for both warmup and cooldown.

    :param max_lr: The peak learning rate.
    :param pct_start: The percentage of total steps used for increasing the LR.
    :param div_factor: Initial LR is max_lr / div_factor; final LR is also reduced similarly.
    :return: Schedule function taking `progress_remaining` and returning the learning rate.
    """
    initial_lr = max_lr / div_factor
    final_lr = max_lr / div_factor

    def schedule(progress_remaining: float) -> float:
        # progress_remaining: 1.0 → 0.0
        step_progress = 1.0 - progress_remaining

        if step_progress < pct_start:
            # Warmup phase (cosine from initial to max)
            warmup_progress = step_progress / pct_start
            lr = initial_lr + 0.5 * (max_lr - initial_lr) * (1 - math.cos(math.pi * warmup_progress))
        else:
            # Cooldown phase (cosine from max to final)
            cooldown_progress = (step_progress - pct_start) / (1 - pct_start)
            lr = max_lr + 0.5 * (final_lr - max_lr) * (1 - math.cos(math.pi * cooldown_progress))
        
        return lr

    return schedule 

def cosine_annealing_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Cosine annealing learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
        current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        min_lr = initial_value * 1e-2
        cosine_decay = 0.5 * (1 + math.cos(math.pi * (1 - progress_remaining)))
        return min_lr + (initial_value - min_lr) * cosine_decay

    return func  
