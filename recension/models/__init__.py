"""Model backends: the provider-agnostic protocol, a deterministic mock, and Anthropic.

``AnthropicModel`` is intentionally not imported here, because importing it requires
the optional ``anthropic`` extra. Import it from
``recension.models.anthropic`` when needed.
"""

from .base import Message, Model, Role
from .mock import MockModel

__all__ = ["Message", "MockModel", "Model", "Role"]
