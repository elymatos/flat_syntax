"""
Flat Syntax Annotation Pipeline

Automatic annotation tool for William Croft's "Flat Syntax" scheme using:
- Trankit parser (Docker service) for Universal Dependencies parsing
- Ollama (local LLM with Llama 3.1:8B) for few-shot disambiguation
- No training required - uses rule-based conversion + few-shot learning
"""

__version__ = "0.1.0"

from flat_syntax.trankit_client import TrankitClient
from flat_syntax.config import PipelineConfig, ServiceConfig, LLMConfig
from flat_syntax.converter import FlatSyntaxConverter
from flat_syntax.formatters import TableFormatter, JSONFormatter, CLDFFormatter
from flat_syntax.models import FlatSyntaxAnnotation

__all__ = [
    "TrankitClient",
    "PipelineConfig",
    "ServiceConfig",
    "LLMConfig",
    "FlatSyntaxConverter",
    "TableFormatter",
    "JSONFormatter",
    "CLDFFormatter",
    "FlatSyntaxAnnotation",
]
