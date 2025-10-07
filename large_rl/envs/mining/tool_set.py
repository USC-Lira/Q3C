import random

import numpy as np
from large_rl.commons.utils import logging


class Tool(object):
    def __init__(self, emb, mine_num_start, mine_num_end):
        self._emb = emb
        self._mns = mine_num_start
        self._mne = mine_num_end

    @property
    def emb(self):
        return self._emb

    @property
    def mns(self):
        return self._mns

    @property
    def mne(self):
        return self._mne


class Tool_set(object):
    def __init__(self, args: dict, prop_to_idx, idx_to_prop):
        self.tools = []
        self._args = args
        self.prop_to_idx = prop_to_idx
        self.idx_to_prop = idx_to_prop
        if not self._args['mw_embedding_perfect']:
            self._embed_mu = 0
            self._embed_sigma = 0.1
        else:
            self._embed_mu = None
            self._embed_sigma = None
        self._rng = np.random.RandomState(self._args["env_seed"])

    def turn_one_hot(self, num, len):
        arr = [0 for _ in range(len + 1)]
        arr[num] += 1
        if not self._args['mw_embedding_perfect']:
            arr = [num + self._rng.normal(self._embed_mu, self._embed_sigma) for num in arr]
        return arr

    def gen_tool_embed(self, node, end_node):

        if self._args['mw_action_id']:
            mine_size = self._args['mw_mine_size']
            embed = [0, 0] + [node * 1.0 / mine_size] + [end_node * 1.0 / mine_size]
        else:
            if self._args['mw_four_dir_actions']:
                if self._args['mw_dir_one_hot']:
                    embed = [0, 0, 0, 0, 0]  # type + direct
                else:
                    embed = [0, 0, 0]  # type + direct
            else:
                embed = [0, 0, 0, 0]  # type + direct
            embed += self.idx_to_prop[node] + self.idx_to_prop[end_node]
        embed = tuple(embed)
        return embed

    def build_tool_set(self, tree):
        """ tree: nested list where each layer contains nodes representing mineId """
        self.tools = []
        tools_dict = dict()  # key: (node, end_node), value: True

        # Associate toos with mines: one tool per mine(node)
        for i, nodes in enumerate(tree):  # Bottom to top
            for node in nodes:
                if i < len(tree) - 1:  # Connect the nodes from l to l-1 layer by Tools
                    end_nodes = tree[i + 1]
                    end_node = self._rng.choice(end_nodes)
                    tools_dict[(node, end_node)] = True
                    tool_embedding = self.gen_tool_embed(node, end_node)
                    tool = Tool(emb=tool_embedding, mine_num_start=node, mine_num_end=end_node)
                else:  # At the top layer, all nodes are going to the root node which is the terminal state
                    tools_dict[(node, self._args["mw_mine_size"])] = True
                    tool_embedding = self.gen_tool_embed(node, self._args["mw_mine_size"])
                    tool = Tool(emb=tool_embedding, mine_num_start=node, mine_num_end=self._args["mw_mine_size"])
                self.tools.append(tool)

        # Build the remaining tools that have not been associated with any mine yet
        # 1. Populate all possible combinations of start and end nodes
        available_tool_num = self._args["mw_tool_size"] - self._args["mw_mine_size"]
        tree_rev = tree[::-1]  # Top to Bottom
        end_nodes = []
        tool_candidates = []
        for i in range(len(tree_rev) - 1):
            end_nodes += tree_rev[i]
            start_nodes = tree_rev[i + 1]
            for e_n in end_nodes:
                for s_n in start_nodes:
                    if tools_dict.get((s_n, e_n)) is None:
                        tool_candidates.append((s_n, e_n))

        # 2. Sample the mine-tool association randomly
        # sample_tools = random.sample(tool_candidates, available_tool_num)
        sample_tools = self._rng.choice(len(tool_candidates), available_tool_num, replace=False)
        for tool in sample_tools:
            start_node, end_node = tool_candidates[tool]
            tool_embedding = self.gen_tool_embed(start_node, end_node)
            new_tool = Tool(emb=tool_embedding, mine_num_start=start_node, mine_num_end=end_node)
            self.tools.append(new_tool)
        logging(f"[Env] Generated {len(self.tools)} tools")

    def get_tool(self, tool_id):
        return self.tools[tool_id]
