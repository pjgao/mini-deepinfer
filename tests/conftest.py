import pytest
import torch

from mini_deepinfer import MiniDeepSeek, ModelConfig

torch.set_num_threads(1)


@pytest.fixture
def model():
    torch.manual_seed(23)
    return MiniDeepSeek(
        ModelConfig(
            vocab_size=32,
            dim=32,
            num_layers=2,
            num_heads=2,
            head_dim=8,
            kv_rank=8,
            rope_dim=4,
            index_dim=8,
            num_experts=3,
            experts_per_token=2,
            expert_dim=32,
            window=3,
            sparse_topk=2,
            max_seq_len=128,
        )
    ).eval()
