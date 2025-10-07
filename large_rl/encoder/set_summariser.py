import torch
import torch.nn as nn


class DeepSet(torch.nn.Module):
    def __init__(self, dim_in: int, dim_out: int, dim_hidden: int = 64, p_drop=0.0,
                 max_pool=False):  # Same as AGILE
        super().__init__()
        self.pre_mean = nn.Sequential(
            nn.Linear(dim_in, dim_hidden), nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden))

        self.post_mean = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden), nn.ReLU(),
            nn.Linear(dim_hidden, dim_out))

        self._p_drop = p_drop
        self._max_pool = max_pool
        if self._p_drop > 0.0:
            self.dropout = nn.Dropout(p=p_drop)

    def forward(self, _in):
        x = self.pre_mean(_in)
        if self._max_pool:
            x = torch.max(x, dim=1)[0]
        else:
            x = torch.mean(x, dim=1)
        x = self.post_mean(x)
        if self._p_drop > 0.0: x = self.dropout(x)
        return x


class BiLSTM(torch.nn.Module):
    def __init__(self, dim_in: int, dim_out: int, dim_hidden: int = 64, num_layers=2, if_summarise=True, p_drop=0.0):
        super().__init__()
        self.dim_hidden = dim_hidden
        self._if_summarise = if_summarise

        self.pre_lstm = nn.Sequential(
            nn.Linear(dim_in, dim_hidden), nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden))

        self.lstm = nn.LSTM(dim_hidden,
                            dim_hidden,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=p_drop,  # If non-zero, introduces a Dropout layer on the outputs
                            bidirectional=True)

        self.post_lstm = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden), nn.ReLU(),
            nn.Linear(dim_hidden, dim_out))

        self._p_drop = p_drop
        if self._p_drop > 0.0:
            self.dropout = nn.Dropout(p=p_drop)

    def forward(self, _in):
        self.lstm.flatten_parameters()
        x = self.pre_lstm(_in)
        x, _ = self.lstm(x)
        x = x.view(x.shape[0], -1, 2, self.dim_hidden)
        if self._if_summarise:
            x = torch.mean(x, dim=[1, 2])  # mean over num-layers and sequence-length
        else:
            x = torch.mean(x, dim=[2])  # mean over num-layers
        x = self.post_lstm(x)
        if self._p_drop > 0.0: x = self.dropout(x)
        return x


class Transformer(torch.nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden=64, num_layers=2, num_heads=1, if_summarise=True, p_drop=0.0):
        super().__init__()
        # there is no batch_first option
        self._if_summarise = if_summarise
        _layer = torch.nn.TransformerEncoderLayer(d_model=dim_in, nhead=num_heads, dim_feedforward=dim_hidden)
        self.model = torch.nn.TransformerEncoder(encoder_layer=_layer, num_layers=num_layers)
        self.post = nn.Sequential(
            nn.Linear(dim_in, dim_hidden), nn.ReLU(),
            nn.Linear(dim_hidden, dim_out)
        )
        self._p_drop = p_drop
        if self._p_drop > 0.0:
            self.dropout = nn.Dropout(p=p_drop)

    def forward(self, _in):
        """ _in: batch x seq x dim """
        _in = _in.permute((1, 0, 2))  # seq x batch x dim
        x = self.model(_in)  # seq x batch x dim
        if self._if_summarise:
            x = torch.mean(x, dim=0)  # batch x dim
        else:
            x = x.permute((1, 0, 2))  # seq x batch x dim -> batch x seq x dim
        x = self.post(x)  # batch x dim_out
        if self._p_drop > 0.0: x = self.dropout(x)
        return x
