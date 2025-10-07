import torch
from torch import nn
from large_rl.commons.pt_activation_fns import ACTIVATION_FN
from large_rl.commons.init_layer import init_layer


class FILM_MLP(nn.Module):
    def __init__(self, dim_in=28, dim_hiddens="256_32", dim_out=1, type_hidden_act_fn=1, if_norm_each=False,
                 if_norm_final=True, if_init_layer=True, in_film=16, p_drop=0.0):
        super(FILM_MLP, self).__init__()
        dim_hiddens = "{}_".format(dim_in) + dim_hiddens
        self._list = dim_hiddens.split("_")
        for i in range(len(self._list) - 1):  # This is Action Encoder
            _in, _out = self._list[i], self._list[i + 1]
            _l = nn.Linear(int(_in), int(_out))
            if if_init_layer: _l = _l.apply(init_layer)
            setattr(self, "dense{}".format(i), _l)
            self.add_module("dense{}".format(i), _l)
            _l = ACTIVATION_FN[type_hidden_act_fn]()
            setattr(self, "act{}".format(i), _l)
            self.add_module("act{}".format(i), _l)
            if if_norm_each and i < len(self._list) - 2:
                _l = LayerNorm(int(_out))
                setattr(self, "norm{}".format(i), _l)
                self.add_module("norm{}".format(i), _l)

            _l = nn.Linear(int(in_film), int(_out))
            if if_init_layer: _l = _l.apply(init_layer)
            setattr(self, "film_gamma{}".format(i), _l)
            self.add_module("film_gamma{}".format(i), _l)

            _l = nn.Linear(int(in_film), int(_out))
            if if_init_layer: _l = _l.apply(init_layer)
            setattr(self, "film_beta{}".format(i), _l)
            self.add_module("film_beta{}".format(i), _l)
        self.norm_out = LayerNorm(int(_out)) if if_norm_final else None
        self.out = nn.Linear(int(_out), dim_out)  # This is Final Q-val computing layer
        nn.init.orthogonal_(self.out.weight, gain=.001)  # more stable
        self._p_drop = p_drop
        if self._p_drop > 0.0:
            self.dropout = nn.Dropout(p=p_drop)

    def forward(self, x, z, if_cache=False):
        for i in range(len(self._list) - 1):
            x = getattr(self, "dense{}".format(i))(x)
            x = getattr(self, "act{}".format(i))(x)
            gamma = getattr(self, "film_gamma{}".format(i))(z)
            beta = getattr(self, "film_beta{}".format(i))(z)
            x = x * gamma + beta
        if self.norm_out is not None:
            x = self.norm_out(x)
        if if_cache: self.input_to_final_layer = x
        x = self.out(x)
        if self._p_drop > 0.0: x = self.dropout(x)
        return x


class MLP(nn.Module):
    def __init__(self, dim_in=28, dim_hiddens="256_32", dim_out=1, type_hidden_act_fn=1, if_norm_each=False,
                 if_norm_final=True, if_init_layer=True, p_drop=0.0):
        super(MLP, self).__init__()
        dim_hiddens = "{}_".format(dim_in) + dim_hiddens
        self._list = dim_hiddens.split("_")
        for i in range(len(self._list) - 1):  # This is Action Encoder
            _in, _out = self._list[i], self._list[i + 1]
            _l = nn.Linear(int(_in), int(_out))
            if if_init_layer: _l = _l.apply(init_layer)
            setattr(self, "dense{}".format(i), _l)
            self.add_module("dense{}".format(i), _l)
            _l = ACTIVATION_FN[type_hidden_act_fn]()
            setattr(self, "act{}".format(i), _l)
            self.add_module("act{}".format(i), _l)
            if if_norm_each and i < len(self._list) - 2:
                _l = LayerNorm(int(_out))
                setattr(self, "norm{}".format(i), _l)
                self.add_module("norm{}".format(i), _l)
        self.norm_out = LayerNorm(int(_out)) if if_norm_final else None
        self.out = nn.Linear(int(_out), dim_out)  # This is Final Q-val computing layer
        nn.init.orthogonal_(self.out.weight, gain=.001)  # more stable
        self._p_drop = p_drop
        if self._p_drop > 0.0:
            self.dropout = nn.Dropout(p=p_drop)

    def forward(self, x, if_cache=False):
        for i in range(len(self._list) - 1):
            x = getattr(self, "dense{}".format(i))(x)
            x = getattr(self, "act{}".format(i))(x)
        if self.norm_out is not None:
            x = self.norm_out(x)
        if if_cache: self.input_to_final_layer = x
        x = self.out(x)
        if self._p_drop > 0.0: x = self.dropout(x)
        return x

    def forward_iterative(self, x):
        """ To deal w/h the CUDA memory error issue """
        # t = torch.cuda.get_device_properties(0).total_memory * 0.000001
        # a = torch.cuda.memory_allocated(0) * 0.000001
        # print(f"total: {t}, allocated: {a}")
        # res = run_cmd(command="nvidia-smi", verbose=True)
        # print(res)
        # _dim, _size = 1, 1000  # split along the item
        _dim, _size = 0, 64  # split along the batch
        batch_in = torch.split(x, split_size_or_sections=_size, dim=_dim)
        _out = [self.forward(x=_in) for _in in batch_in]
        out = torch.cat(_out, dim=_dim)
        return out


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self._gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(dim=1).view(*shape)
        std = x.view(x.size(0), -1).std(dim=1).view(*shape)

        y = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] if x.dim() == 2 else [1, 1, -1]
            # shape = [1, -1] + [1] * (x.dim() - 2)
            y = self._gamma.view(*shape) * y + self.beta.view(*shape)
        return y


def _test_NeuralNet():
    """ test method """
    print("=== _test_NeuralNet ===")
    batch_size, num_items, dim_item = 7, 19, 10
    model = MLP(dim_in=dim_item, if_norm_each=True)
    print(model)
    x = torch.randn(batch_size, num_items, dim_item)
    out = model(x)
    print(f"out: {out.shape}")
    out = model.forward_iterative(x)
    print(f"out: {out.shape}")


if __name__ == '__main__':
    _test_NeuralNet()
