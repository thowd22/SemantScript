"""Teacher backend implementations."""

from semantscript_trainer.teachers.anthropic import AnthropicTeacher, BatchHandle
from semantscript_trainer.teachers.ollama import DEFAULT_OLLAMA_BASE_URL, OllamaTeacher

__all__ = [
    "DEFAULT_OLLAMA_BASE_URL",
    "AnthropicTeacher",
    "BatchHandle",
    "OllamaTeacher",
]
