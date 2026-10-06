"""Compatibility module for the Brunno llama.cpp server presets.

The actively maintained AgentPresets implementation lives in init_model.py.
Keeping this module preserves the original project layout while avoiding two
independent implementations of the same server-management logic.
"""

from .init_model import AgentPresets

__all__ = ["AgentPresets"]
