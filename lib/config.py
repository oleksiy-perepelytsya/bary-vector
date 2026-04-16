from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, default))


def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, default))


def _env_bool(key: str, default: bool = False) -> bool:
    v = os.environ.get(key)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


# Default q_seed per L14 edge_type (fermion order). See CLAUDE.md "Stage 5".
# Overridable individually via Q_SEED_<UPPER>.
_DEFAULT_Q_SEEDS: dict[str, float] = {
    "contradicts": 0.85,
    "applies_to": 0.55,
    "is_instance_of": 0.65,
    "extends": 0.60,
    "same_phenomenon": 0.70,
    "synonyms": 0.90,
}


def _load_q_seeds() -> dict[str, float]:
    return {k: _env_float(f"Q_SEED_{k.upper()}", v) for k, v in _DEFAULT_Q_SEEDS.items()}


@dataclass(frozen=True)
class Settings:
    # --- Mongo ---
    mongo_uri: str = "mongodb://localhost:27117/?directConnection=true"
    mongo_db: str = "barygraph_poc"
    mongo_collection: str = "barygraph"
    # Safety net for integration-test teardown: only DBs starting with this prefix
    # may be dropped by the test fixture. Changing this requires updating the
    # fixture guard in tests/conftest.py.
    mongo_test_db_prefix: str = "barygraph_test_"

    # --- Models (Ollama) ---
    ollama_url: str = "http://localhost:11434"
    embed_model: str = "nomic-embed-text:v1.5"
    llm_model: str = "llama4-scout:q4_k_m"
    embed_dim: int = 768
    embed_timeout_seconds: float = 120.0
    llm_timeout_seconds: float = 300.0
    llm_max_tokens: int = 80
    fake_embed: bool = False
    fake_llm: bool = False

    # --- Data / state ---
    kaikki_path: Path = field(default_factory=lambda: Path("data/kaikki-en.jsonl"))
    parsed_dir: Path = field(default_factory=lambda: Path("data/parsed"))
    pipeline_state_dir: Path = field(default_factory=lambda: Path("pipeline_state"))

    # --- Batching ---
    batch_size: int = 512          # generic ingest / bulk-write batch
    embed_batch_size: int = 64     # tokens-per-request cap for the embedder

    # --- Tunable thresholds (pipeline) ---
    q_min_l15: float = 0.72                    # Stage 3b greedy-match floor
    disambig_threshold: float = 0.72           # lib.disambiguate cosine fallback
    meta_bary_cos_threshold: float = 0.90      # Stage 7: L13 triad formation
    polysemy_q_floor: float = 0.40             # open question; tune post L14
    summary_token_limit: int = 60              # Stage 8: reject if exceeded
    # Stage 8 "needsLLM" strength cutoffs (see CLAUDE.md):
    summary_strong_cutoff: float = 0.85        # same_phenomenon / extends
    summary_applies_to_cutoff: float = 0.80

    # --- L14 edge q_seeds (fermion order) ---
    q_seeds: dict[str, float] = field(default_factory=_load_q_seeds)

    # --- Misc ---
    log_level: str = "INFO"

    @classmethod
    def load(cls, dotenv_path: str | os.PathLike | None = None) -> Settings:
        """Load settings from environment, with optional .env file."""
        load_dotenv(dotenv_path, override=False)
        return cls(
            mongo_uri=_env_str("MONGO_URI", cls.mongo_uri),
            mongo_db=_env_str("MONGO_DB", cls.mongo_db),
            mongo_collection=_env_str("MONGO_COLLECTION", cls.mongo_collection),
            mongo_test_db_prefix=_env_str("MONGO_TEST_DB_PREFIX", cls.mongo_test_db_prefix),
            ollama_url=_env_str("OLLAMA_URL", cls.ollama_url),
            embed_model=_env_str("EMBED_MODEL", cls.embed_model),
            llm_model=_env_str("LLM_MODEL", cls.llm_model),
            embed_dim=_env_int("EMBED_DIM", cls.embed_dim),
            embed_timeout_seconds=_env_float("EMBED_TIMEOUT_SECONDS", cls.embed_timeout_seconds),
            llm_timeout_seconds=_env_float("LLM_TIMEOUT_SECONDS", cls.llm_timeout_seconds),
            llm_max_tokens=_env_int("LLM_MAX_TOKENS", cls.llm_max_tokens),
            fake_embed=_env_bool("BARY_FAKE_EMBED", False),
            fake_llm=_env_bool("BARY_FAKE_LLM", False),
            kaikki_path=Path(_env_str("KAIKKI_PATH", "data/kaikki-en.jsonl")),
            parsed_dir=Path(_env_str("PARSED_DIR", "data/parsed")),
            pipeline_state_dir=Path(_env_str("PIPELINE_STATE_DIR", "pipeline_state")),
            batch_size=_env_int("BATCH_SIZE", cls.batch_size),
            embed_batch_size=_env_int("EMBED_BATCH_SIZE", cls.embed_batch_size),
            q_min_l15=_env_float("Q_MIN_L15", cls.q_min_l15),
            disambig_threshold=_env_float("DISAMBIG_THRESHOLD", cls.disambig_threshold),
            meta_bary_cos_threshold=_env_float(
                "META_BARY_COS_THRESHOLD", cls.meta_bary_cos_threshold
            ),
            polysemy_q_floor=_env_float("POLYSEMY_Q_FLOOR", cls.polysemy_q_floor),
            summary_token_limit=_env_int("SUMMARY_TOKEN_LIMIT", cls.summary_token_limit),
            summary_strong_cutoff=_env_float(
                "SUMMARY_STRONG_CUTOFF", cls.summary_strong_cutoff
            ),
            summary_applies_to_cutoff=_env_float(
                "SUMMARY_APPLIES_TO_CUTOFF", cls.summary_applies_to_cutoff
            ),
            q_seeds=_load_q_seeds(),
            log_level=_env_str("LOG_LEVEL", cls.log_level),
        )
