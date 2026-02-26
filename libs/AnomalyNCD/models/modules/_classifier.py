import torch
import torch.nn as nn

"""
The code is based on SimGCD and UNO, the source code is available at:
https://github.com/DonkeyShot21/UNO/blob/main/utils/nets.py (UNO)
https://github.com/CVMI-Lab/SimGCD/blob/main/model.py (SimGCD)
"""

class MultiHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, 
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256, n_head=4):
        super().__init__()
        self.num_head = n_head

        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        elif nlayers != 0:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights) 
        self.last_layer = nn.ModuleList([nn.utils.weight_norm(nn.Linear(in_dim, out_dim, bias=False)) for _ in range(n_head)])
        for i in range(n_head):
            self.last_layer[i].weight_g.data.fill_(1)

        if norm_last_layer:
            for i in range(n_head):
                self.last_layer[i].weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x_proj = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        logits = [self.last_layer[i](x) for i in range(self.num_head)]

        return x_proj, logits


def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]