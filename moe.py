import torch
import torch.nn as nn
import torch.nn.functional as F


class MoEGate(nn.Module):
    def __init__(self, dim, num_experts):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, num_experts),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class MoE(nn.Module):
    def __init__(self, dim, num_experts):
        super().__init__()
        self.num_experts = num_experts
        self.k = 2
        self.gating_function = MoEGate(dim, num_experts)
        scaling_factor = 4
        self.expert_weights_1 = nn.Parameter(
            torch.empty(self.num_experts, dim * scaling_factor, dim)
        )
        self.expert_weights_2 = nn.Parameter(
            torch.empty(self.num_experts, dim * scaling_factor, dim)
        )

        self.expert_biases_1 = nn.Parameter(torch.zeros(self.num_experts, dim, 1))
        self.expert_biases_2 = nn.Parameter(torch.zeros(self.num_experts, dim, 1))

        nn.init.xavier_normal_(self.expert_weights_1)
        nn.init.xavier_normal_(self.expert_weights_2)

    def forward(self, x: torch.Tensor):
        b, s, d = x.shape
        gate_output: torch.Tensor = self.gating_function(x)

        values, indices = gate_output.topk(self.k)
        values = values / torch.sum(values, dim=-1, keepdim=True)

        ffn_weights_1 = self.expert_weights_1[indices]
        ffn_weights_2 = self.expert_weights_2[indices]

        ffn_biases_1 = self.expert_biases_1[indices]
        ffn_biases_2 = self.expert_biases_2[indices]

        x = x.unsqueeze(-1).unsqueeze(2)

        x = F.gelu((ffn_weights_1 @ x) + ffn_biases_1)
        x = F.gelu((ffn_weights_2 @ x) + ffn_biases_2)

        x = x.view(b, s, self.k, d)
        x = x * values.unsqueeze(-1)
        return torch.sum(x, dim=-2)
