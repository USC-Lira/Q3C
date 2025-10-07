import numpy as np
import torch
from torch import nn

from large_rl.policy.arch.mlp import MLP, FILM_MLP
from large_rl.commons.init_layer import init_layer_noise, init_layer_add_noise
from large_rl.encoder.set_summariser import DeepSet, BiLSTM, Transformer
from large_rl.encoder.rnn import RNNFamily


def non_parametric_list_encoding(slate, _type):
    if _type == "moments":  # Original CDQN does this!
        _a = torch.cat([slate.mean(axis=1), slate.std(axis=1)], dim=-1)
        _a[torch.isnan(_a)] = 0.0  # replace nan with 0
    elif _type == "concat":  # horizontally concat list elements
        batch, list_len, dim = slate.shape
        _a = slate.view(batch, list_len * dim)
    else:
        raise ValueError
    return _a


def _get_non_parametric_dim_slate(_index, dim_action, _type):
    if _type == "moments":
        return dim_action * 2
    elif _type == "concat":
        _index = 1 if _index < 2 else _index
        return dim_action * _index


class CellCritic(nn.Module):
    """ Auto-regressive Critic """

    def __init__(self, dim_state, dim_memory, dim_action, dim_out=1, num_layers=2, args=None):
        super(CellCritic, self).__init__()
        self._args = args
        self._dim_memory = dim_memory
        self._num_layers = num_layers

        self._if_list_one_hot = self._args["WOLP_ar_type_listwise_update"] == "next-list-index" and self._args[
            "WOLP_cascade_type_list_reward"] == "last"
        if self._if_list_one_hot:
            self._list_one_hot = torch.eye(n=self._args["WOLP_cascade_list_len"], device=self._args["device"],
                                           requires_grad=False)

        if self._args["WOLP_if_ar_critic_cascade"]:
            if self._if_list_one_hot:
                _dim_in = dim_state + dim_action + self._args["WOLP_cascade_list_len"]
            else:
                _dim_in = dim_state + dim_action
            if self._args["WOLP_if_film_listwise"] and self._args["WOLP_if_ar_cascade_list_enc"] and dim_memory != 0:
                self.Q_net = FILM_MLP(dim_in=_dim_in,
                                      dim_hiddens=args["Qnet_dim_hidden"],
                                      dim_out=dim_out,
                                      if_norm_each=self._args["WOLP_if_critic_norm_each"],
                                      if_norm_final=self._args["WOLP_if_critic_norm_final"],
                                      if_init_layer=self._args["WOLP_if_critic_init_layer"],
                                      type_hidden_act_fn=10,
                                      in_film=dim_memory
                                      ).to(device=args["device"])
            else:
                if self._args["WOLP_if_ar_cascade_list_enc"]: _dim_in += dim_memory
                self.Q_net = MLP(dim_in=_dim_in,
                                 dim_hiddens=args["Qnet_dim_hidden"],
                                 dim_out=dim_out,
                                 if_norm_each=self._args["WOLP_if_critic_norm_each"],
                                 if_norm_final=self._args["WOLP_if_critic_norm_final"],
                                 if_init_layer=self._args["WOLP_if_critic_init_layer"],
                                 type_hidden_act_fn=10).to(device=args["device"])
            self.cell = None
        else:
            if self._args["WOLP_type_ar_critic_GRU"].lower() == "both":
                _dim_in_gru = dim_state + dim_action
                _dim_in_q = self._dim_memory
            elif self._args["WOLP_type_ar_critic_GRU"].lower() == "state":
                _dim_in_gru = dim_state
                _dim_in_q = self._dim_memory + dim_action
            elif self._args["WOLP_type_ar_critic_GRU"].lower() == "action":
                _dim_in_gru = dim_action
                _dim_in_q = self._dim_memory + dim_state
            else:
                raise ValueError

            if self._args["WOLP_ar_type_cell"].lower() == "gru":
                self.cell = nn.GRU(input_size=_dim_in_gru,
                                   hidden_size=self._dim_memory,
                                   batch_first=True,
                                   num_layers=self._num_layers)
            elif self._args["WOLP_ar_type_cell"].lower() == "lstm":
                self.cell = nn.LSTM(input_size=_dim_in_gru,
                                    hidden_size=self._dim_memory,
                                    batch_first=True,
                                    num_layers=self._num_layers)
            elif self._args["WOLP_ar_type_cell"].lower() == "rnn":
                self.cell = nn.RNN(input_size=_dim_in_gru,
                                   hidden_size=self._dim_memory,
                                   batch_first=True,
                                   num_layers=self._num_layers)

            if self._args["WOLP_if_ar_full_listEnc"]:
                _dim_in_q += dim_memory

            if self._args["WOLP_if_ar_critic_use_prevAction"]:
                assert not self._args["WOLP_if_ar_full_listEnc"], "list-embed and prev-action can't coexist"
                _dim_in_q += dim_action

            self.Q_net = MLP(
                dim_in=_dim_in_q,
                dim_hiddens=args["Qnet_dim_hidden"],
                dim_out=dim_out,
                if_norm_each=self._args["WOLP_if_critic_norm_each"],
                if_norm_final=self._args["WOLP_if_critic_norm_final"],
                if_init_layer=self._args["WOLP_if_critic_init_layer"],
                type_hidden_act_fn=10).to(device=args["device"])

    def forward(self, state, list_embed, prev_action, action, memory, **kwargs):
        if self._if_list_one_hot:
            _one_hot = self._list_one_hot[kwargs["list_index"]][None, None, :].repeat(state.shape[0], 1, 1)
        if not self._args["WOLP_if_ar_critic_cascade"]:
            if self._args["WOLP_type_ar_critic_GRU"].lower() == "both":
                out, memory = self.cell(torch.cat([state, action], dim=-1), memory)
                return self.Q_net(out).squeeze(-1), memory
            elif self._args["WOLP_type_ar_critic_GRU"].lower() == "state":
                state, memory = self.cell(state, memory)
            elif self._args["WOLP_type_ar_critic_GRU"].lower() == "action":
                action, memory = self.cell(action, memory)
            else:
                raise ValueError

        if list_embed is not None:
            if self._args["WOLP_if_film_listwise"]:
                if self._if_list_one_hot:
                    # one-hot: batch x 1 x dim
                    _input = torch.cat([state, action, _one_hot], dim=-1).squeeze(1)
                else:
                    # state, list-embed, action: batch x 1 x dim
                    _input = torch.cat([state, action], dim=-1).squeeze(1)
                return self.Q_net(_input, list_embed.squeeze(1)), memory
            else:
                if self._if_list_one_hot:
                    # one-hot: batch x 1 x dim
                    _input = torch.cat([state, list_embed, action, _one_hot], dim=-1).squeeze(1)
                else:
                    # state, list-embed, action: batch x 1 x dim
                    _input = torch.cat([state, list_embed, action], dim=-1).squeeze(1)
        else:
            if prev_action is not None:
                _input = torch.cat([state, prev_action, action], dim=-1).squeeze(1)
            else:
                _input = torch.cat([state, action], dim=-1).squeeze(1)
        return self.Q_net(_input), memory


class ARCritic(nn.Module):
    """ Auto-regressive Critic """

    def __init__(self, dim_state, dim_hidden, dim_memory, dim_action, num_layers=1, args=None):
        super(ARCritic, self).__init__()
        self._args = args
        self._dim_memory = dim_memory
        self._num_layers = num_layers
        self._if_share = self._args["WOLP_if_ar_critic_share_weight"]

        assert not np.alltrue([self._args["WOLP_if_ar_full_listEnc"], self._args["WOLP_if_ar_critic_cascade"]])
        if self._args["WOLP_ar_type_list_encoder"].lower() in ["concat", "moment"]:
            assert not self._args["WOLP_if_ar_critic_share_weight"], "We need different dim for each list-index"

        if self._args["WOLP_if_ar_critic_share_weight"]:
            self.cells = [CellCritic(dim_state=dim_state, dim_memory=dim_memory, num_layers=self._num_layers,
                                     dim_action=dim_action, args=args)]
        else:
            self.cells = list()
            for i in range(self._args["WOLP_cascade_list_len"]):
                if i == 0 and self._args["WOLP_t0_no_list_input"]:
                    dim_memory = 0
                elif self._args["WOLP_ar_type_list_encoder"].lower() in ["concat", "moment"]:
                    dim_memory = _get_non_parametric_dim_slate(_index=i, dim_action=dim_action,
                                                               _type=args["WOLP_ar_type_list_encoder"].lower())
                else:
                    dim_memory = self._dim_memory
                _cell = CellCritic(dim_state=dim_state, dim_memory=dim_memory, num_layers=self._num_layers,
                                   dim_action=dim_action, args=args)
                if self._args["WOLP_ar_critic_type_init_weight"] == "random":
                    _cell = _cell.apply(init_layer_noise)
                elif self._args["WOLP_ar_critic_type_init_weight"] == "add":
                    _cell = _cell.apply(init_layer_add_noise)
                self.cells.append(_cell)

        for i, head in enumerate(self.cells):
            self.add_module(f"cell_{i}", head)

        self.list_encoder = None
        if self._args["WOLP_if_ar_full_listEnc"] or (self._args["WOLP_if_ar_critic_cascade"] and
                                                     self._args["WOLP_if_ar_cascade_list_enc"]):
            if self._args["WOLP_list_concat_state"]:
                dim_list = dim_state + dim_action
            else:
                dim_list = dim_action
            if self._args["WOLP_ar_type_list_encoder"].lower() == "transformer":
                self.list_encoder = Transformer(dim_in=dim_list, dim_out=dim_memory, dim_hidden=dim_hidden)
            elif self._args["WOLP_ar_type_list_encoder"].lower() == "lstm":
                self.list_encoder = BiLSTM(dim_in=dim_list, dim_out=dim_memory, dim_hidden=dim_hidden)
            elif self._args["WOLP_ar_type_list_encoder"].lower() == "deepset":
                self.list_encoder = DeepSet(dim_in=dim_list, dim_out=dim_memory, dim_hidden=dim_hidden,
                                            max_pool = self._args["WOLP_ar_list_encoder_deepset_maxpool"])
            elif self._args["WOLP_ar_type_list_encoder"].lower() in ["concat", "moment"]:
                self.list_encoder = lambda x: x  # temp
            elif self._args["WOLP_ar_type_list_encoder"].lower().startswith("non-shared"):
                _type = self._args["WOLP_ar_type_list_encoder"].lower().split("non-shared-")[-1]
                self._type = _type
                self.list_encoder = list()
                for i in range(self._args["WOLP_cascade_list_len"]):
                    if i == 0 and self._args["WOLP_t0_no_list_input"]:
                        l = None
                    elif _type == "linear":
                        _dim_in = _get_non_parametric_dim_slate(_index=i, dim_action=dim_action, _type="concat")
                        _dim_in += dim_state * self._args["WOLP_list_concat_state"]
                        l = nn.Sequential(nn.Linear(_dim_in, dim_hidden), nn.ReLU(), nn.Linear(dim_hidden, dim_memory))
                    elif _type == "transformer":
                        l = Transformer(dim_in=dim_list, dim_out=dim_memory, dim_hidden=dim_hidden)
                    elif _type == "lstm":
                        l = BiLSTM(dim_in=dim_list, dim_out=dim_memory, dim_hidden=dim_hidden)
                    elif _type == "deepset":
                        l = DeepSet(dim_in=dim_list, dim_out=dim_memory, dim_hidden=dim_hidden,
                                    max_pool=self._args["WOLP_ar_list_encoder_deepset_maxpool"])
                    self.list_encoder.append(l)
                    self.add_module("list_enc_{}".format(i), l)
            elif self._args["WOLP_ar_type_list_encoder"].lower() == "rnn":
                self.list_encoder = RNNFamily(dim_in=dim_list, dim_out=dim_memory, dim_hidden=dim_hidden,
                                              device=self._args["device"])
            else:
                raise ValueError

    def forward(self, state, action_seq, true_action=None, if_next_list_Q=False,
                knn_function=None,
                alternate_conditioning=None,
                start_index=0, **kwargs):
        if not self._args["WOLP_if_ar_critic_cascade"]: [_model.cell.flatten_parameters() for _model in self.cells]

        if self._args["WOLP_ar_type_cell"] == "lstm":
            memory = (torch.zeros(self._num_layers, state.shape[0], self._dim_memory, device=self._args["device"]),
                      torch.zeros(self._num_layers, state.shape[0], self._dim_memory, device=self._args["device"]))
        else:
            memory = torch.zeros(self._num_layers, state.shape[0], self._dim_memory, device=self._args["device"])

        q = list()
        list_embed, prev_action = None, None
        for t in range(start_index, self._args["WOLP_cascade_list_len"]):
            if self.list_encoder is not None:
                if t == 0:
                    _a = torch.zeros(state.shape[0], 1, action_seq.shape[-1], device=self._args["device"])
                else:
                    if alternate_conditioning is not None:
                        _a = alternate_conditioning[:, :t, :]
                    elif knn_function is not None:
                        _a = knn_function(action_seq.detach()[:, :1, :]) if t == 1 else\
                            torch.cat([_a, knn_function(action_seq.detach()[:, t-1:t, :])], dim=1)
                    elif self._args["WOLP_if_ar_detach_list_action"]:
                        _a = action_seq[:, :t, :].detach()
                    else:
                        _a = action_seq[:, :t, :]
                if self._args["WOLP_if_ar_full_listEnc"] or self._args["WOLP_if_ar_critic_cascade"]:
                    # When non-shared, we can skip the list_encoder at t=0
                    if t == 0 and self._args["WOLP_t0_no_list_input"]:
                        list_embed = None
                    elif self._args["WOLP_ar_type_list_encoder"].lower() in ["concat", "moment"]:
                        list_embed = non_parametric_list_encoding(slate=_a, _type=self._args[
                            "WOLP_ar_type_list_encoder"].lower()).unsqueeze(1)
                    elif self._args["WOLP_ar_type_list_encoder"].lower().startswith("non-shared"):
                        if self._type == "linear":
                            _input = non_parametric_list_encoding(slate=_a, _type="concat")
                            if self._args["WOLP_list_concat_state"]:
                                _input = torch.cat([state.squeeze(1), _input], dim=-1)
                        else:
                            _input = _a
                            if self._args["WOLP_list_concat_state"]:
                                _input = torch.cat([state.repeat(1, _input.shape[1], 1), _input], dim=-1)
                        list_embed = self.list_encoder[t](_input).unsqueeze(1)
                    else:
                        if self._args["WOLP_list_concat_state"]:
                            list_embed = self.list_encoder(torch.cat([state.repeat(1, _a.shape, 1), _a], dim=-1)).unsqueeze(1)
                        else:
                            list_embed = self.list_encoder(_a).unsqueeze(1)
            else:  # if contextual-prop or full-prop
                if self._args["WOLP_if_ar_critic_use_prevAction"]:
                    if t == 0:
                        prev_action = torch.zeros(state.shape[0], 1, action_seq.shape[-1], device=self._args["device"])
                    else:
                        prev_action = action_seq[:, t - 1, :].unsqueeze(1)
            _q, memory = self.cells[0 if self._if_share else t](state=state,
                                                                list_embed=list_embed,
                                                                prev_action=prev_action,
                                                                action=action_seq[:, t, :].unsqueeze(1) if true_action is None else true_action,
                                                                memory=memory, list_index=t)
            if if_next_list_Q: return _q
            q.append(_q)
        return torch.stack(q, dim=1).squeeze(-1)


class CellActor(nn.Module):
    def __init__(self, dim_in, dim_memory, dim_out, args=None, num_layers=2):
        super(CellActor, self).__init__()
        self._args = args
        self._dim_out = dim_out
        self._num_layers = num_layers

        assert not np.alltrue([self._args["WOLP_if_ar_actor_use_prevAction"], self._args["WOLP_if_ar_actor_cascade"]])

        self._if_list_one_hot = self._args["WOLP_ar_type_listwise_update"] == "next-list-index" and self._args[
            "WOLP_cascade_type_list_reward"] == "last"

        if self._if_list_one_hot:
            self._list_one_hot = torch.eye(n=self._args["WOLP_cascade_list_len"], device=self._args["device"],
                                           requires_grad=False)

        if self._args["WOLP_if_ar_actor_use_prevAction"]:
            _in = dim_in + dim_out  # Same as cddpg concatenating [state, slate] as input
        elif self._args["WOLP_if_ar_actor_cascade"] and self._args["WOLP_if_ar_cascade_list_enc"]:
            _in = dim_in + dim_memory  # Same as cddpg concatenating [state, slate] as input
        else:
            _in = dim_in

        if self._args["WOLP_if_ar_actor_cascade"]:
            if self._if_list_one_hot: _in += self._args["WOLP_cascade_list_len"]
            self.cell = None
            # import pudb; pudb.start()
            if self._args["WOLP_if_film_listwise"] and self._args["WOLP_if_ar_cascade_list_enc"] and dim_memory != 0:
                self.model = FILM_MLP(dim_in=_in - dim_memory,  # need to subtract because added above
                                      dim_out=dim_out,
                                      type_hidden_act_fn=10,
                                      dim_hiddens=self._args["WOLP_actor_dim_hiddens"],
                                      if_init_layer=self._args["WOLP_if_actor_init_layer"],
                                      if_norm_each=self._args["WOLP_if_actor_norm_each"],
                                      if_norm_final=self._args["WOLP_if_actor_norm_final"],
                                      in_film=dim_memory, )
            else:
                self.model = MLP(dim_in=_in,
                                 dim_out=dim_out,
                                 type_hidden_act_fn=10,
                                 dim_hiddens=self._args["WOLP_actor_dim_hiddens"],
                                 if_init_layer=self._args["WOLP_if_actor_init_layer"],
                                 if_norm_each=self._args["WOLP_if_actor_norm_each"],
                                 if_norm_final=self._args["WOLP_if_actor_norm_final"], )
        else:
            if self._args["WOLP_ar_type_cell"].lower() == "gru":
                self.cell = nn.GRU(input_size=_in, hidden_size=dim_memory, batch_first=True,
                                   num_layers=self._num_layers)
            elif self._args["WOLP_ar_type_cell"].lower() == "lstm":
                self.cell = nn.LSTM(input_size=_in, hidden_size=dim_memory, batch_first=True,
                                    num_layers=self._num_layers)
            elif self._args["WOLP_ar_type_cell"].lower() == "rnn":
                self.cell = nn.RNN(input_size=_in, hidden_size=dim_memory, batch_first=True,
                                   num_layers=self._num_layers)

            if self._args["WOLP_if_ar_full_listEnc"] and (not self._args["WOLP_if_ar_actor_cascade"]):
                # NOTE: NOT USED
                dim_memory *= 2
            self.model = MLP(dim_in=dim_memory,
                             dim_out=dim_out,
                             type_hidden_act_fn=10,
                             dim_hiddens=self._args["WOLP_actor_dim_hiddens"],
                             if_init_layer=self._args["WOLP_if_actor_init_layer"],
                             if_norm_each=self._args["WOLP_if_actor_norm_each"],
                             if_norm_final=self._args["WOLP_if_actor_norm_final"], )

    def forward(self, state, list_embed, prev_action, memory, **kwargs):
        if self._if_list_one_hot:
            _one_hot = self._list_one_hot[kwargs["list_index"]][None, None, :].repeat(state.shape[0], 1, 1)
        x = torch.cat([state, prev_action[:, None, :]], dim=-1) if prev_action is not None else state

        if self.cell is not None:
            x, memory = self.cell(x, memory)

        if list_embed is not None and not self._args["WOLP_if_film_listwise"]:
            x = torch.cat([x, list_embed.unsqueeze(1)], dim=-1)

        if self._if_list_one_hot:
            x = torch.cat([x, _one_hot], dim=-1)

        if self._args["WOLP_if_film_listwise"] and list_embed is not None:
            x = self.model(x, list_embed.unsqueeze(1)).squeeze(1)
        else:
            x = self.model(x).squeeze(1)

        if self._args["DEBUG_type_activation"].lower() == "tanh":
            out = torch.tanh(x)
            out = out * self._args["env_max_action"]
        elif self._args["DEBUG_type_activation"].lower() == "sigmoid":
            out = torch.sigmoid(x)
        return out, memory


class ARActor(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_memory, dim_out, args=None, num_layers=1):
        super(ARActor, self).__init__()
        self._args = args
        if self._args["WOLP_ar_type_list_encoder"] in ["concat", "moment"]:
            dim_memory = dim_out
        self._dim_memory = dim_memory
        self._dim_out = dim_out
        self._num_layers = num_layers

        assert not np.alltrue([self._args["WOLP_if_ar_full_listEnc"], self._args["WOLP_if_ar_actor_cascade"]])
        if self._args["WOLP_ar_type_list_encoder"].lower() in ["concat", "moment"]:
            assert not self._args["WOLP_if_ar_actor_share_weight"], "We need different dim for each list-index"

        if self._args["WOLP_if_ar_actor_share_weight"]:
            self.cells = [CellActor(dim_in=dim_in, dim_memory=dim_memory, dim_out=dim_out, num_layers=self._num_layers,
                                    args=args)]
        else:
            self.cells = list()
            # import pudb; pudb.start()
            for i in range(self._args["WOLP_cascade_list_len"]):
                if i == 0 and self._args["WOLP_t0_no_list_input"]:
                    dim_memory = 0
                elif self._args["WOLP_ar_type_list_encoder"].lower() in ["concat", "moment"]:
                    dim_memory = _get_non_parametric_dim_slate(_index=i, dim_action=dim_out,
                                                               _type=args["WOLP_ar_type_list_encoder"].lower())
                else:
                    dim_memory = self._dim_memory
                _cell = CellActor(dim_in=dim_in, dim_memory=dim_memory, dim_out=dim_out, num_layers=self._num_layers,
                                  args=args)
                if self._args["WOLP_ar_actor_type_init_weight"] == "random":
                    _cell = _cell.apply(init_layer_noise)
                elif self._args["WOLP_ar_actor_type_init_weight"] == "add":
                    _cell = _cell.apply(init_layer_add_noise)
                self.cells.append(_cell)

        for i, head in enumerate(self.cells):
            self.add_module(f"cell_{i}", head)

        self.list_encoder = None
        # import pudb; pudb.start()
        if self._args["WOLP_if_ar_full_listEnc"] or (self._args["WOLP_if_ar_actor_cascade"] and
                                                     self._args["WOLP_if_ar_cascade_list_enc"]):
            if self._args["WOLP_list_concat_state"]:
                dim_list = dim_in + dim_out
            else:
                dim_list = dim_out
            if self._args["WOLP_ar_type_list_encoder"].lower() == "transformer":
                self.list_encoder = Transformer(dim_in=dim_list, dim_out=dim_memory, dim_hidden=dim_hidden)
            elif self._args["WOLP_ar_type_list_encoder"].lower() == "lstm":
                self.list_encoder = BiLSTM(dim_in=dim_list, dim_out=dim_memory, dim_hidden=dim_hidden)
            elif self._args["WOLP_ar_type_list_encoder"].lower() == "deepset":
                self.list_encoder = DeepSet(dim_in=dim_list, dim_out=dim_memory, dim_hidden=dim_hidden,
                                            max_pool=self._args["WOLP_ar_list_encoder_deepset_maxpool"])
            elif self._args["WOLP_ar_type_list_encoder"].lower() in ["concat", "moment"]:
                self.list_encoder = lambda x: x  # temp
            elif self._args["WOLP_ar_type_list_encoder"].lower().startswith("non-shared"):
                _type = self._args["WOLP_ar_type_list_encoder"].lower().split("non-shared-")[-1]
                self._type = _type
                self.list_encoder = list()
                for i in range(self._args["WOLP_cascade_list_len"]):
                    if i == 0 and self._args["WOLP_t0_no_list_input"]:
                        l = None
                    elif _type == "linear":
                        _dim_in = _get_non_parametric_dim_slate(_index=i, dim_action=dim_out, _type="concat")
                        _dim_in += dim_in * self._args["WOLP_list_concat_state"]
                        l = nn.Sequential(nn.Linear(_dim_in, dim_hidden), nn.ReLU(), nn.Linear(dim_hidden, dim_memory))
                    elif _type == "transformer":
                        l = Transformer(dim_in=dim_list, dim_out=dim_memory, dim_hidden=dim_hidden)
                    elif _type == "lstm":
                        l = BiLSTM(dim_in=dim_list, dim_out=dim_memory, dim_hidden=dim_hidden)
                    elif _type == "deepset":
                        l = DeepSet(dim_in=dim_list, dim_out=dim_memory, dim_hidden=dim_hidden,
                                    max_pool=self._args["WOLP_ar_list_encoder_deepset_maxpool"])
                    self.list_encoder.append(l)
                    self.add_module("list_enc_{}".format(i), l)
            elif self._args["WOLP_ar_type_list_encoder"].lower() == "rnn":
                self.list_encoder = RNNFamily(dim_in=dim_list, dim_out=dim_memory, dim_hidden=dim_hidden,
                                              device=self._args["device"])

    def forward(self, state, if_next_list_Q=False, eps=None, knn_function=None,
                alternate_conditioning=None, **kwargs):
        if not self._args["WOLP_if_ar_actor_cascade"]: [_model.cell.flatten_parameters() for _model in self.cells]

        if self._args["WOLP_ar_type_cell"].lower() == "lstm":
            memory = (torch.zeros(self._num_layers, state.shape[0], self._dim_memory, device=self._args["device"]),
                      torch.zeros(self._num_layers, state.shape[0], self._dim_memory, device=self._args["device"]))
        else:
            memory = torch.zeros(self._num_layers, state.shape[0], self._dim_memory, device=self._args["device"])
        knn_action = torch.zeros(state.shape[0], 1, self._dim_out, device=self._args["device"])
        list_embed, prev_action = None, None
        for t in range(self._args["WOLP_cascade_list_len"]):
            if alternate_conditioning is not None:
                list_action = alternate_conditioning[:, :t, :]
            elif self._args["WOLP_if_ar_detach_list_action"]:
                list_action = knn_action.detach()
            else:
                list_action = knn_action
            if self.list_encoder is not None:
                if t == 0 and self._args["WOLP_t0_no_list_input"]:
                    list_embed = None
                elif self._args["WOLP_ar_type_list_encoder"].lower() in ["concat", "moment"]:
                    list_embed = non_parametric_list_encoding(slate=list_action,
                                                              _type=self._args["WOLP_ar_type_list_encoder"].lower())
                elif self._args["WOLP_ar_type_list_encoder"].lower().startswith("non-shared"):
                    if self._type == "linear":
                        _input = non_parametric_list_encoding(slate=list_action, _type="concat")
                        if self._args["WOLP_list_concat_state"]:
                            _input = torch.cat([state.squeeze(1), _input], dim=-1)
                    else:
                        _input = list_action
                        if self._args["WOLP_list_concat_state"]:
                            _input = torch.cat([state.repeat(1, _input.shape[1], 1), _input], dim=-1)
                    list_embed = self.list_encoder[t](_input).squeeze(1)
                else:
                    if self._args["WOLP_list_concat_state"]:
                        list_embed = self.list_encoder(torch.cat([state(1, list_action.shape[1], 1), list_action], dim=-1))
                    else:
                        list_embed = self.list_encoder(list_action)
                # if "_flg" in kwargs:
                #     if kwargs["_flg"]: print("actor", t, list_embed[0].abs().sum())

            if (not self._args["WOLP_if_ar_actor_cascade"]) and self._args["WOLP_if_ar_actor_use_prevAction"]:
                prev_action = list_action[-1] if t > 0 else list_action
            _action, memory = self.cells[0 if self._args["WOLP_if_ar_actor_share_weight"] else t](
                state=state, list_embed=list_embed, prev_action=prev_action, memory=memory, list_index=t
            )
            # Add noise here, so that the cascaded actors are evaluated on noisy actions,
            # which would actually go to the Q-function
            if self._args["WOLP_if_ar_noise_before_cascade"] and not if_next_list_Q and eps is not None:
                _action += eps[:, t, :]
                if self._args["DEBUG_type_clamp"] == "small":
                    _action = _action.clamp(0, 1)  # Constraint the range
                elif self._args["DEBUG_type_clamp"] == "large":
                    _action = _action.clamp(-1, 1)  # Constraint the range

            _knn_action = knn_function(_action.unsqueeze(1)) if knn_function is not None else _action[:, None, :]
            if t == 0:
                action = _action[:, None, :]
                knn_action = _knn_action
            else:
                action = torch.cat([action, _action[:, None, :]], dim=1)
                knn_action = torch.cat([knn_action, _knn_action], dim=1)
            if if_next_list_Q: return action
        return action


def _test():
    from large_rl.commons.seeds import set_randomSeed
    set_randomSeed()
    dim_state, dim_hidden, dim_action, dim_memory, batch_size = 5, 6, 5, 7, 3
    args = {
        "device": "cpu",
        "WOLP_cascade_list_len": 4,
        "Qnet_dim_hidden": "32_16",
        "WOLP_actor_dim_hiddens": "32_16",
        "WOLP_if_ar_actor_use_prevAction": False,
        # "WOLP_if_ar_actor_use_prevAction": True,
        "WOLP_if_ar_critic_use_prevAction": False,
        # "WOLP_if_ar_critic_use_prevAction": True,
        "WOLP_if_ar_actor_share_weight": False,
        # "WOLP_if_ar_actor_share_weight": True,
        "WOLP_if_ar_critic_share_weight": False,
        # "WOLP_if_ar_critic_share_weight": True,
        "agent_action_encoder_layers": "64",
        "agent_action_encoder_dim_out": 32,
        "WOLP_if_ar_actor_cascade": True,
        # "WOLP_if_ar_actor_cascade": False,
        "WOLP_if_ar_critic_cascade": True,
        # "WOLP_if_ar_critic_cascade": False,
        # "WOLP_if_ar_full_listEnc": True,
        "WOLP_if_ar_full_listEnc": False,
        # "WOLP_type_ar_critic_GRU": "both",
        # "WOLP_type_ar_critic_GRU": "action",
        "WOLP_type_ar_critic_GRU": "state",
        # "WOLP_ar_type_list_encoder": "transformer",
        # "WOLP_ar_type_list_encoder": "lstm",
        # "WOLP_ar_type_list_encoder": "concat",
        # "WOLP_ar_type_list_encoder": "non-shared-linear",
        # "WOLP_ar_type_list_encoder": "non-shared-deepset",
        "WOLP_ar_type_list_encoder": "rnn",
        "DEBUG_type_activation": "tanh",
        "WOLP_ar_type_cell": "rnn",
        "WOLP_if_critic_norm_each": True,
        "WOLP_if_actor_norm_each": True,
        "WOLP_if_critic_norm_final": True,
        "WOLP_if_actor_norm_final": True,
        "WOLP_if_critic_init_layer": True,
        "WOLP_if_actor_init_layer": True,
        "WOLP_if_ar_cascade_list_enc": True,
        # "WOLP_if_ar_cascade_list_enc": False,
        "WOLP_ar_actor_type_init_weight": "random",
        "WOLP_ar_critic_type_init_weight": "random",
        "WOLP_if_ar_detach_list_action": False,
        "WOLP_if_ar_noise_before_cascade": False,
        "WOLP_ar_type_listwise_update": "next-list-index",
        "WOLP_cascade_type_list_reward": "last",
    }
    state = torch.randn(batch_size, 1, dim_state)
    m = ARActor(dim_in=dim_state, dim_hidden=dim_hidden, dim_memory=dim_memory, dim_out=dim_action, args=args)
    m.eval()
    print(m.training)
    action_seq = m(state=state, _flg=True)
    print(action_seq.shape)

    m = ARCritic(dim_state=dim_state, dim_hidden=dim_hidden, dim_memory=dim_memory, dim_action=dim_action, args=args)
    m.eval()
    print(m.training)
    q = m(state=state, action_seq=action_seq, _flg=True)
    print(q.shape)


if __name__ == '__main__':
    _test()
