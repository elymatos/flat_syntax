"""
Configuration classes for Flat Syntax pipeline.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class ServiceConfig:
    """Configuration for Trankit parser service."""

    # Service endpoint
    url: str = "http://localhost:8406"

    # Default language
    language: str = "english"

    # Timeout settings
    timeout: int = 30

    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0

    # Batch processing
    batch_size: int = 100

    @classmethod
    def from_env(cls):
        """Load configuration from environment variables."""
        return cls(
            url=os.getenv('TRANKIT_SERVICE_URL', cls.__dataclass_fields__['url'].default),
            language=os.getenv('TRANKIT_LANGUAGE', cls.__dataclass_fields__['language'].default),
            timeout=int(os.getenv('TRANKIT_TIMEOUT', cls.__dataclass_fields__['timeout'].default)),
        )


@dataclass
class LLMConfig:
    """Configuration for LLM client."""

    provider: str = "ollama"  # ollama (default), openai, anthropic
    model: str = "llama3.1:8b"  # Default Ollama model
    base_url: str = "http://localhost:11434"  # Ollama service URL
    api_key: Optional[str] = None  # Only for OpenAI/Anthropic
    temperature: float = 0
    max_tokens: int = 500


@dataclass
class PipelineConfig:
    """Configuration for complete pipeline."""

    # Trankit service
    trankit_url: str = "http://localhost:8406"
    language: str = "english"

    # LLM refinement
    use_llm_refinement: bool = True
    llm_config: Optional[LLMConfig] = None
    few_shot_path: Optional[str] = None

    # Output
    output_format: str = "json"  # json, csv, table
    output_path: Optional[str] = None

    # Processing
    batch_size: int = 100
    num_workers: int = 1

    def __post_init__(self):
        """Initialize default LLM config if not provided."""
        if self.llm_config is None and self.use_llm_refinement:
            self.llm_config = LLMConfig()

    @classmethod
    def from_yaml(cls, path: str):
        """Load configuration from YAML file."""
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)

        # Handle nested LLMConfig
        if 'llm_config' in data and isinstance(data['llm_config'], dict):
            data['llm_config'] = LLMConfig(**data['llm_config'])

        return cls(**data)
