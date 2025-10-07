""" Used to define a pytorch based neural network """
from torch import nn
import numpy as np

ACTIVATION_FN = {
    1: nn.ELU,
    2: nn.Hardshrink,
    4: nn.Hardtanh,
    6: nn.LeakyReLU,
    7: nn.LogSigmoid,
    8: nn.MultiheadAttention,
    9: nn.PReLU,
    10: nn.ReLU,
    11: nn.ReLU6,
    12: nn.RReLU,
    13: nn.SELU,
    14: nn.CELU,
    16: nn.Sigmoid,
    18: nn.Softplus,
    19: nn.Softshrink,
    20: nn.Softsign,
    21: nn.Tanh,
    22: nn.Tanhshrink,
    23: nn.Threshold,
    24: nn.Softmin,
    25: nn.Softmax,
    26: nn.Softmax2d,
    27: nn.LogSoftmax,
    28: nn.AdaptiveLogSoftmaxWithLoss,
    29: nn.Identity
}


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


# Maps behaviour into ctr - organic has real support ctr is on [0,1].
def ff(xx, aa=5, bb=2, cc=0.3, dd=2, ee=6):
    # Magic numbers give a reasonable ctr of around 2%.
    return sig(aa * sig(bb * sig(cc * xx) - dd) - ee)
