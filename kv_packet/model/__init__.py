from transformers import (
    GraniteForCausalLM,
    LlamaForCausalLM,
    Qwen3ForCausalLM,
)
from typing import TypeAlias

SupportedModel: TypeAlias = \
    GraniteForCausalLM | \
    LlamaForCausalLM | \
    Qwen3ForCausalLM
