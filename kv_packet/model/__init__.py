from transformers import (
    LlamaForCausalLM,
    Qwen3ForCausalLM,
)
from lm_engine.hf_models.models.energy import EnergyForCausalLM
from typing import TypeAlias

SupportedModel: TypeAlias = \
    LlamaForCausalLM | \
    Qwen3ForCausalLM | \
    EnergyForCausalLM


def is_energy_model(model: SupportedModel) -> bool:
    return model.config.model_type == "energy"
