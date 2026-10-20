import pytest

from mini_deepinfer import ModelConfig, SamplingParams


@pytest.mark.parametrize("value", [1.5, float("nan"), True])
def test_model_dimensions_require_integers(value):
    with pytest.raises(TypeError, match="num_layers"):
        ModelConfig(num_layers=value)


@pytest.mark.parametrize("name", ["max_tokens", "top_k"])
@pytest.mark.parametrize("value", [1.5, float("nan"), True])
def test_sampling_counts_require_integers(name, value):
    with pytest.raises(TypeError, match=name):
        SamplingParams(**{name: value})
