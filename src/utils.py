"""config/config.yml -> typed config; .env loading; store construction."""

import os

import yaml

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from qdrant_client.models import Datatype, Memory

from .qdrant_store import QdrantEmbeddingStore, QdrantPreferences

ROOT_DIR = Path(__file__).resolve().parent.parent

API_KEY_ENV_VARS = ("OPENAI_API_KEY", "VLM_API_KEY")


@dataclass
class ExtractionConfig:
    col_extraction_model: str = "Metric-AI/ColQwen2.5-3b-multilingual-v1.0"
    device: str = "cpu"
    reports_dir: Path = Path("data/reports")
    pages_dir: Path = Path("data/pages")
    pdf_image_dpi: int = 150
    embed_batch_size: int = 2
    top_k: int = 5


@dataclass
class VLMConfig:
    model: str
    max_tokens: int = 2048
    timeout: float = 180.0
    max_retries: int = 4
    concurrency: int = 4
    image_format: str = "jpeg"

    @property
    def api_key(self) -> str:
        """From the environment only, never from config.yml."""
        for name in API_KEY_ENV_VARS:
            if os.environ.get(name):
                return os.environ[name]
        raise RuntimeError(f"no API key: set one of {' or '.join(API_KEY_ENV_VARS)} in .env")


@dataclass
class OutputConfig:
    output_dir: Path = Path("data/output")


@dataclass
class QdrantConfig:
    url: str = "http://localhost:6333"
    collection_name: str = "report_pages"
    prefer_grpc: bool = True
    timeout: int | None = 120
    datatype: str = "float16"
    memory: str = "cold"
    on_disk_payload: bool = True
    hnsw_m: int = 0


@dataclass
class AppConfig:
    extraction: ExtractionConfig
    vlm: VLMConfig
    output: OutputConfig
    qdrant: QdrantConfig

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly copy, for run_config_<run_id>.json."""
        return {name: {k: str(v) if isinstance(v, Path) else v for k, v in asdict(getattr(self, name)).items()}
                for name in ("extraction", "vlm", "output", "qdrant")}


def _build(cls, raw: dict[str, Any] | None):
    """Dataclass from a YAML section: unknown keys ignored, Path fields resolved against the repo root."""
    path_fields = {f.name for f in fields(cls) if f.type in (Path, "Path")}
    known = {f.name for f in fields(cls)}
    config = cls(**{k: v for k, v in (raw or {}).items() if k in known})
    for name in path_fields:
        value = Path(getattr(config, name))
        setattr(config, name, value if value.is_absolute() else ROOT_DIR / value)
    return config


def load_config(config_path: Path | None = None) -> AppConfig:
    with open(config_path or ROOT_DIR / "config" / "config.yml", "r") as f:
        raw = yaml.safe_load(f) or {}
    return AppConfig(
        extraction=_build(ExtractionConfig, raw.get("extraction")),
        vlm=_build(VLMConfig, raw.get("vlm")),
        output=_build(OutputConfig, raw.get("output")),
        qdrant=_build(QdrantConfig, raw.get("qdrant")),
    )


def load_env(path: Path | None = None) -> None:
    """KEY=value lines from .env into the environment; existing variables win."""
    env_path = path or ROOT_DIR / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def store_from_config(config: QdrantConfig) -> QdrantEmbeddingStore:
    """`distance` is not configurable: MaxSim over ColQwen assumes cosine."""
    preferences = QdrantPreferences(
        prefer_grpc=config.prefer_grpc,
        datatype=Datatype(config.datatype),
        memory=Memory(config.memory),
        hnsw_m=config.hnsw_m,
        on_disk_payload=config.on_disk_payload,
        timeout=config.timeout,
    )

    return QdrantEmbeddingStore(collection_name=config.collection_name,
                                url=config.url,
                                api_key=os.environ.get("QDRANT_API_KEY"),
                                qdrant_preferences=preferences)
