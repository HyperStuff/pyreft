import math

import numpy as np
import torch
import torch.nn.functional as F


class ScaledDotProductAttention(torch.nn.Module):
    """
    Adapted from: https://github.com/sooftware/attentions/blob/master/attentions.py
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values

    Args: dim, mask
        dim (int): dimention of attention
        mask (torch.Tensor): tensor containing indices to be masked

    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection vector for decoder.
        - **key** (batch, k_len, d_model): tensor containing projection vector for encoder.
        - **value** (batch, v_len, d_model): tensor containing features of the encoded input sequence.
        - **mask** (-): tensor containing indices to be masked

    Returns: context, attn
        - **context**: tensor containing the context vector from attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the encoder outputs.
    """

    def __init__(self, dim: int, dtype: torch.dtype):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)
        self.dtype = dtype
        self.W_q = torch.nn.Linear(dim, dim, bias=False, dtype=dtype)
        self.W_k = torch.nn.Linear(dim, dim, bias=False, dtype=dtype)
        self.W_v = torch.nn.Linear(dim, dim, bias=False, dtype=dtype)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        query = self.W_q(query)
        key = self.W_k(key)
        value = self.W_v(value)

        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim

        if mask is not None:
            score.masked_fill_(mask.view(score.size()), -float("Inf"))

        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context, attn


class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0.5).to(input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)


class DiscreteTokenSelection(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int,
        start_temperature: float,
        end_temperature: int,
        total_steps: int,
        dtype: torch.dtype = torch.float32,
        scheduler: str = "linear",
        discretization_strategy: str = "default_sigmoid",
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.start_temperature = start_temperature
        self.end_temperature = end_temperature
        self.total_steps = total_steps
        self.dtype = dtype
        self.scheduler = scheduler
        self.discretization_strategy = discretization_strategy

        self.temperature = torch.nn.Parameter(
            torch.ones(1, 1, 1, dtype=dtype) * self.start_temperature,
            requires_grad=False,
        )

        self.layernorm = torch.nn.LayerNorm(self.embed_dim, dtype=dtype)
        self.down_proj = torch.nn.Linear(
            self.embed_dim,
            2 if self.discretization_strategy == "binary_entropy" else 1,
            dtype=dtype,
        )
        self._current_step = 0

        self.register_backward_hook(self._update_temperature)

    def get_temperature(self) -> float:
        if self.discretization_strategy not in ["default_sigmoid", "binary_concrete"]:
            return None
        return self.temperature.item()

    def _update_temperature(self, *args):
        self._current_step = min(self._current_step + 1, self.total_steps)

        if self.scheduler == "linear":
            current_temp = self.start_temperature + (
                self.end_temperature - self.start_temperature
            ) * (self._current_step / self.total_steps)
        elif self.scheduler == "cosine":
            current_temp = self.start_temperature + (
                self.end_temperature - self.start_temperature
            ) * 0.5 * (1 + math.cos(math.pi * self._current_step / self.total_steps))
        elif self.scheduler == "sigmoid":
            current_temp = self.start_temperature + (
                self.end_temperature - self.start_temperature
            ) * 1 / (1 + math.exp(-12 * (self._current_step / self.total_steps - 0.5)))
        else:
            raise ValueError(f"Invalid scheduler: {self.scheduler}")

        self.temperature.data.fill_(current_temp)

    def forward(self, x: torch.Tensor, eps=1e-8) -> torch.Tensor:
        """Uses bidirectional self-attention to compute token weights."""
        if x.shape[-1] != 1:
            x = self.layernorm(x)
            x = self.down_proj(x)

        if self.discretization_strategy == "default_sigmoid":
            out = F.sigmoid(x / self.temperature)
        elif self.discretization_strategy == "ste":
            return STEFunction.apply(x)
        elif self.discretization_strategy == "binary_concrete":
            uniform_sample = torch.rand_like(x)
            logistic_noise = torch.log(uniform_sample + eps) - torch.log(
                1 - uniform_sample + eps
            )
            out = F.sigmoid((x + logistic_noise) / self.temperature)
        elif self.discretization_strategy == "single_entropy":
            out = F.sigmoid(x)
        elif self.discretization_strategy == "binary_entropy":
            # just take last dimension of (B, S, 2)
            out = F.sigmoid(x)[:, :, -1, None]
        else:
            raise NotImplementedError

        return out


def compute_entropy_loss(x: torch.Tensor, mode="single", eps=1e-8) -> torch.Tensor:
    """Compute vector-based and scalar entropy."""

    def logeps(x):
        return torch.log(x + eps)

    if mode == "single":
        return (-x * logeps(x) - (1 - x) * logeps(1 - x)).mean()
    else:
        return -torch.sum(x * logeps(x), dim=-1).mean()
