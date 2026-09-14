"""The extraction graph, one run per ingested report:
embed_queries --Send per year--> answer_year --> collect.

One VLM call per year of the grid (10 per report), each covering the 4 scopes of
that year. The questions are constants in `questions.py`; ColQwen is only used to
encode the 10 retrieval queries, so the parallel branches never touch the GPU.
"""

from __future__ import annotations

import logging
import operator

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, TypedDict, Literal

import numpy as np

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from langgraph.runtime import Runtime

from .embedding_store import EmbeddingStore, SearchHit
from .embedding import Embedder
from .parsing import Record, records_from
from .pdf import page_image_path
from .questions import SCOPES, SYSTEM_PROMPT, YEARS, retrieval_query, get_user_prompt
from .vlm import VLMClient, VLMResponse

logger = logging.getLogger(__name__)


@dataclass
class Context:
    embedder: Embedder
    embedding_store: EmbeddingStore
    ai_client: VLMClient | None
    top_k: int
    is_dry_run: bool


@dataclass(frozen=True)
class Hit:
    page: int
    score: float


@dataclass
class Retrieval:
    """Formatted data after query embedding store"""
    year: int
    query: str
    hits: list[Hit]

    @property
    def pages(self) -> list[int]:
        return [h.page for h in self.hits]


@dataclass
class VLMCall:
    year: int
    pages_sent: list[int]
    scores: list[float]
    # dry run: retrieval happened, the model was not called
    skipped: bool = False
    response: VLMResponse | None = None
    # failure before the model was called
    error: str | None = None
    n_positives: int = 0


class ReportState(TypedDict, total=True):
    # input, set by the extract command
    doc_id: str         # qdrant_tenant
    report_name: str    # name.pdf
    images_dir: Path
    page_count: int

    # embed_queries
    query_embeddings: dict[int, np.ndarray]

    # answer_year: each branch appends
    retrievals: Annotated[list[Retrieval], operator.add]
    calls: Annotated[list[VLMCall], operator.add]
    records: Annotated[list[Record], operator.add]

    # collect
    rows: Annotated[list[ExtractedRow], operator.add]


@dataclass
class ExtractedRow:
    """Row Record for comparision with the gold standard

    => report_name, report_year, year, scope, value, unity,
    unity_normalized (unit_normalization_dict.csv), page, display_type,
    extracted_text_from_page"""
    report_name: str              # From us
    sent_pages: list[int] | None  # From retriver (Precisa adpatar já que são 5)

    llm_year: str                               # From LLM
    llm_scope: Literal["1", "2mb", "2lb", "3"]  # From LLM
    llm_value: float | None                     # From LLM
    llm_unit: str | None                        # From LLM
    raw_json_from_vlm: str | None               # From LLM


class YearJob(TypedDict):
    """What one `Send` carries to `answer_year`."""
    year: int
    query_embedding: np.ndarray
    doc_id: str
    report_name: str
    images_dir: Path

## UTIL ---
def _page_of(hit: SearchHit) -> int:
    """The `page` payload, else parsed from an id like "page_093"."""
    page = hit.metadata.get("page")
    return int(page) if page is not None else int(str(hit.id).removeprefix("page_"))


## NODES ---
def embed_queries_(state: ReportState, runtime: Runtime[Context]) -> dict[str, Any]:
    queries = [retrieval_query(y) for y in YEARS]
    embeddings = runtime.context.embedder.embed_queries(queries)
    return {
        "query_embeddings": dict(zip(YEARS, embeddings))
    }


def fan_out_(state: ReportState) -> list[Send]:
    return [
        Send("answer_year",
                YearJob(year=year,
                        query_embedding=state.get("query_embeddings")[year], # type: ignore
                        doc_id=state.get("doc_id"),                          # type: ignore
                        report_name=state.get("report_name"),                # type: ignore
                        images_dir=Path(state.get("images_dir")))            # type: ignore
            )
        for year in YEARS
    ]


def answer_year_(state: YearJob, runtime: Runtime[Context]) -> dict[str, Any]:
    """
    Adds to graph state the VLM call result:
    ```json
    {
        "retrievals": list[Retrieval(year, str_query, list[Hit(int_page, score)])],
        "calls": VLMCall,
        "records": list[Record]
    }
    ```
    """
    year = state.get("year")

    # Get most relevant pages (top-k) from eval document
    tenant_id = state.get("doc_id")
    embeded_query = state.get("query_embedding")
    hits = runtime.context.embedding_store.search(tenant_id,
                                                  embeded_query,
                                                  limit=runtime.context.top_k)
    retrieval_data = Retrieval(year,
                               retrieval_query(year),
                               [Hit(_page_of(h), float(h.score)) for h in hits])

    # Prepare VLM
    pages = retrieval_data.pages
    ai_call = VLMCall(year,
                      pages_sent=pages,
                      scores=[h.score for h in retrieval_data.hits])

    result: dict[str, Any] = {
        "retrievals": [retrieval_data],
        "calls": [ai_call],
        "records": []
    }

    # If its dry run, we'll return just the retrival data
    if runtime.context.is_dry_run or runtime.context.ai_client is None:
        ai_call.skipped = True
        return result

    if not pages:
        ai_call.error = "no pages retrieved (empty tenant?)"
        return result

    # Get images path
    images = [page_image_path(state.get("images_dir"), p) for p in pages]
    missing = [str(p) for p in images if not p.exists()]
    if missing:
        ai_call.error = f"page image(s) missing: {missing[:3]}"
        return result

    # Call VLM
    response = runtime.context.ai_client.ask(images,
                                             SYSTEM_PROMPT,
                                             get_user_prompt(year))  # type: ignore[union-attr]

    records = records_from(response.parsed_response, year) if response.parsed_response else []
    ai_call.response = response
    ai_call.n_positives = sum(1 for r in records if r.value is not None)

    if response.error is None:
        logger.info("%s %d: pages %s -> %d positive(s), tokens %s in / %s out, %.1fs",
                    state.get("report_name"),
                    year,
                    pages,
                    ai_call.n_positives,
                    response.prompt_tokens,
                    response.completion_tokens,
                    response.latency_s)

    result["records"] = records
    return result


def collect_(state: ReportState, runtime: Runtime[Context]) -> dict[str, Any]:
    report_name = state.get("report_name")
    retrievals_by_year = {r.year: r for r in state.get("retrievals", [])}

    # Sorted records by scope and year
    # 1/2023, 2mb/2023, 2lb/2023, 3/2023...
    records = sorted(state.get("records", []),
                     key=lambda r: (list(SCOPES).index(r.scope), r.year))

    # Rows of extracted data to csv
    extracted_rows: list[ExtractedRow] = []

    for r in records:
        extracted_rows.append(ExtractedRow(
            report_name=report_name,
            sent_pages=retrievals_by_year[r.year].pages,
            llm_year=str(r.year),
            llm_scope=r.scope,
            llm_value=r.value,
            llm_unit=r.unit,
            raw_json_from_vlm=None
        ))

    return { "rows": extracted_rows }


def build_graph():
    builder = StateGraph(state_schema=ReportState,
                         context_schema=Context,
                         output_schema=ReportState)

    builder.add_node("embed_queries", embed_queries_)
    builder.add_node("answer_year", answer_year_, input_schema=YearJob)
    builder.add_node("collect", collect_)

    builder.add_edge(START, "embed_queries")
    builder.add_conditional_edges("embed_queries", fan_out_, ["answer_year"])
    builder.add_edge("answer_year", "collect")
    builder.add_edge("collect", END)

    return builder.compile(name="extraction")
