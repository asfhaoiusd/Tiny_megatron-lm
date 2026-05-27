import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class MQAconfig:
    n_layers: int = 12
    n_heads: int = 12
    d_model: int = 768
    d_kv: int = 64
    d_ff: int = 3072
    n_kv_heads: int = 1
    n_kv_groups: int = 1

    def __post_init__(self) -> None:
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.n_kv_groups = self.n_heads // self.n_kv_heads

class MQA(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_kv = config.d_kv
        self.d_ff = config.d_ff
        self.n_kv_heads = config.n_kv_heads
        self.n_kv_groups = config.n_kv_groups
        self.n_rep = self.n_heads // self.n_kv_heads

        self.q_proj = nn.Linear(self.d_model, self.d_kv * self.n_heads)
        self.k_proj = nn.Linear(self.d_model, self.d_kv * self.n_kv_heads)
        self.v_proj = nn.Linear(self.d_model, self.d_kv * self.n_kv_heads)
        self.o_proj = nn.Linear(self.d_kv * self.n_heads, self.d_model)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_rep == 1:
            return x
        b, n_kv, t, d = x.shape
        return x[:, :, None, :, :].expand(b, n_kv, self.n_rep, t, d).reshape(b, n_kv * self.n_rep, t, d)

    def forward(self, x, past_key_values=None, use_cache=False):
        batch_size, seq_len, _ = x.shape

        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)

        query = query.view(batch_size, seq_len, self.n_heads, self.d_kv).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.n_kv_heads, self.d_kv).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.n_kv_heads, self.d_kv).transpose(1, 2)

        if past_key_values is not None:
            past_k, past_v = past_key_values
            key = torch.cat([past_k, key], dim=2)
            value = torch.cat([past_v, value], dim=2)

        key = self._repeat_kv(key)
        value = self._repeat_kv(value)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_kv)
        scores = torch.softmax(scores, dim=-1)

        output = torch.matmul(scores, value)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_kv * self.n_heads)
        output = self.o_proj(output)

        return output, (key, value)

def main():
    config = MQAconfig(
        n_layers=12,
        n_heads=12,
        d_model=768,
        d_kv=64,
        d_ff=3072,
        n_kv_heads=12,
        n_kv_groups=1,
    )
    model = MQA(config)
    x = torch.randn(1, 10, 768)
    output, (key, value) = model(x)
    print(output.shape)
    print(key.shape)
    print(value.shape)

if __name__ == "__main__":
    main()

