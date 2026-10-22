"""Project-wide settings, read from environment variables and an optional `.env`.

Why a config module?
    Every later step needs knobs: which Whisper model to load, which LLM to call,
    where to write outputs. Hard-coding them scatters magic strings across the
    code. Instead, all knobs live in ONE typed class below, and every module asks
    `get_settings()` for them.

How values are resolved (highest priority first):
    1. Real environment variables, e.g. `export NOTETAKER_WHISPER_MODEL=base`
    2. A `.env` file in the current folder (copy `.env.example` to start)
    3. The defaults written in this file

pydantic-settings does the reading and also *validates* each value against its
type hint. For example `whisper_model: Literal["tiny", ...]` means a typo like
`NOTETAKER_WHISPER_MODEL=smal` fails loudly at startup with a clear message,
instead of crashing 20 minutes into a transcription.

Secrets (API keys) use `SecretStr`, which prints as `**********` so a key never
leaks into logs or error messages by accident. The defaults are all free and
local: nothing here requires a paid API key.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Every variable name starts with this, so ours never clash with other tools'
# variables (many programs read a generic `MODEL` or `DEVICE`, for instance).
ENV_PREFIX = "NOTETAKER_"

# Type aliases for the allowed values. `Literal[...]` means "exactly one of
# these strings"; pydantic rejects anything else.
WhisperModel = Literal["tiny", "base", "small", "medium", "large-v3"]
Device = Literal["auto", "cpu", "cuda"]
ComputeType = Literal["auto", "int8", "float16", "float32"]
LLMProvider = Literal["ollama", "groq", "huggingface"]


class Settings(BaseSettings):
    """All configurable settings, grouped by the roadmap phase that uses them.

    Each field becomes an environment variable named NOTETAKER_<FIELD_NAME>
    (case does not matter), e.g. `whisper_model` <- `NOTETAKER_WHISPER_MODEL`.
    """

    # `model_config` is how pydantic classes are configured (the name is
    # pydantic's, nothing to do with ML models).
    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_file=".env",
        env_file_encoding="utf-8",
        # Ignore unrelated lines in .env instead of erroring on them.
        extra="ignore",
    )

    # ---- Paths ----------------------------------------------------------
    data_dir: Path = Field(
        default=Path("data"),
        description="Where downloaded lectures and eval sets live (gitignored).",
    )
    output_dir: Path = Field(
        default=Path("outputs"),
        description="Where transcripts, notes, and flashcards are written.",
    )

    # ---- Phase 1: transcription (faster-whisper) ------------------------
    whisper_model: WhisperModel = Field(
        default="small",
        description="Whisper model size. Bigger is more accurate but slower.",
    )
    whisper_device: Device = Field(
        default="auto",
        description="'auto' uses a GPU if one is available, else the CPU.",
    )
    whisper_compute_type: ComputeType = Field(
        default="auto",
        description="Numeric precision. int8 is fastest on CPU.",
    )
    language: str | None = Field(
        default="en",
        description="Spoken language code, or empty to let Whisper detect it.",
    )
    sample_rate: int = Field(
        default=16_000,
        gt=0,  # "greater than 0": pydantic rejects 0 or negative numbers.
        description="Audio is resampled to this rate (Hz); Whisper expects 16 kHz.",
    )

    # ---- Phase 3: notes generation (LLM) --------------------------------
    llm_provider: LLMProvider = Field(
        default="ollama",
        description="ollama = local and free; groq / huggingface = free hosted tiers.",
    )
    llm_model: str = Field(
        default="llama3.1:8b",
        description="Model name as the chosen provider spells it.",
    )
    llm_temperature: float = Field(
        default=0.2,
        ge=0.0,  # "greater or equal": temperature must be in [0, 2].
        le=2.0,
        description="Randomness of LLM output. Low values give steadier notes.",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Address of the local Ollama server.",
    )
    # API keys also accept the names the providers' own docs use (GROQ_API_KEY,
    # HF_TOKEN). `AliasChoices` lists every accepted variable name, first wins.
    groq_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(f"{ENV_PREFIX}GROQ_API_KEY", "GROQ_API_KEY"),
        description="Free-tier Groq key. Only needed when llm_provider=groq.",
    )
    hf_token: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(f"{ENV_PREFIX}HF_TOKEN", "HF_TOKEN"),
        description="Free Hugging Face token. Only for llm_provider=huggingface.",
    )

    # ---- Phase 5: search across lectures (RAG) --------------------------
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Small, free sentence-transformers model for embeddings.",
    )

    # A "validator" is a hook pydantic runs on a value before storing it.
    # Environment variables are always strings, so a blank line such as
    # `NOTETAKER_LANGUAGE=` gives "" rather than None. This turns blank values
    # into None, meaning "auto-detect" for language and "no key" for API keys
    # (so the blank keys in .env.example count as unset, not as an empty key).
    @field_validator("language", "groq_api_key", "hf_token", mode="before")
    @classmethod
    def _blank_means_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    def redacted(self) -> dict[str, object]:
        """Return settings as a plain dict that is safe to print.

        `model_dump()` turns the object into a dict. SecretStr values stay
        masked, and unset keys show as None, so this never reveals a secret.
        """
        dumped = self.model_dump()
        return {
            key: ("**********" if isinstance(value, SecretStr) else value)
            for key, value in dumped.items()
        }


# `lru_cache` remembers the result of the first call and returns that same
# object afterwards. So the environment and .env are read once per process,
# and every module shares one Settings object (a common "singleton" pattern).
# Tests call `get_settings.cache_clear()` to force a fresh read.
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once and reuse them everywhere."""
    return Settings()
