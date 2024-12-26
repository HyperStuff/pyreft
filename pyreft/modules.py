import torch


class TokenSelectionAttention(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        start_temperature: float,
        end_temperature: int,
        total_steps: int,
        dropout: float = 0.0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.start_temperature = start_temperature
        self.end_temperature = end_temperature
        self.total_steps = total_steps
        self.dtype = dtype

        self.temperature = torch.nn.Parameter(
            torch.ones(1, 1, 1, dtype=dtype) * self.start_temperature,
            requires_grad=False,
        )

        self.attn = torch.nn.MultiheadAttention(
            self.embed_dim, self.num_heads, dropout=dropout, dtype=dtype
        )
        self.down_proj = torch.nn.Linear(self.embed_dim, 1, dtype=dtype)
        self._current_step = 0

        self.register_backward_hook(self._update_temperature)

    def _update_temperature(self, *args):
        self._current_step = min(self._current_step + 1, self.total_steps)
        current_temp = self.start_temperature + (
            self.end_temperature - self.start_temperature
        ) * (self._current_step / self.total_steps)
        self.temperature.data.fill_(current_temp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(x, x, x)
        selection_mask = torch.nn.functional.sigmoid(
            self.down_proj(out) / self.temperature
        )
        return selection_mask
