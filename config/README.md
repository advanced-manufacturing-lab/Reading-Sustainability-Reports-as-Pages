# config.yml

The four sections of `config.yml` are loaded by `load_config()` in `src/utils.py`, each
into a typed dataclass. Unknown keys are silently ignored, and path fields are resolved
against the repository root, so a relative path here always points to the same place no
matter where the command is run from.

The VLM API key does **not** live here: it comes from `.env`, as either `OPENAI_API_KEY`
or `VLM_API_KEY`.

## Extraction

Settings for the deterministic side of the pipeline — page rendering, the embedding
model, and retrieval. They are read both by `ingest.py`, which turns the PDFs into
images and indexes the vectors, and by `extract.py`, which reuses the same model to
embed the retrieval queries. The model and the device must match across the two
commands: vectors produced by one model are not comparable to those of another.

* `col_extraction_model`: the ColQwen used to embed pages (during ingest) and retrieval
  queries (during extract). It is multi-vector — a page becomes a matrix of per-patch
  vectors rather than a single vector, and that is what Qdrant's MaxSim search compares.
* `device`: where ColQwen runs — `cuda`, `mps` or `cpu`. On `mps` and `cpu` the cache is
  emptied between batches; this is the bottleneck of the ingest.
* `reports_dir`: folder holding the input PDFs. Only `ingest.py` reads from it, picking
  up every top-level `*.pdf`.
* `pages_dir`: where the permanent page PNGs are written, as
  `<pages_dir>/<PDF stem>/page_NNN.png`, plus one `manifest.json` per report.
  `extract.py` discovers what there is to extract by scanning those manifests, not
  `reports_dir`.
* `pdf_image_dpi`: rendering resolution of the pages, capped by `MAX_IMAGE_SIDE` in
  `src/pdf.py`: a page whose longest side would exceed it is rendered at a lower dpi
  instead. Without that cap a landscape page at 150 dpi is 22k patches for ColQwen, and
  the attention over those does not fit in memory. The same PNGs are what the VLM sees.
* `embed_batch_size`: how many pages ColQwen embeds at a time. Controls peak GPU memory;
  2 is conservative for 150 dpi.
* `top_k`: how many pages each search returns. This is also how many images go into
  every VLM call, so it drives both the cost per call and the risk of the right page
  being left out.

## VLM

Settings for the vision model and the pipeline's only network call. Read only by
`extract.py`, and not always: under `--dry-run` the client is never constructed, so
nothing in this section is used and no API key is required. There are ten calls per
report, one per year of the grid, each covering that year's four scopes.

* `model`: the provider's model identifier, in the form the OpenAI-compatible API
  expects.
* `max_tokens`: ceiling on response tokens. If the model hits the limit the response
  comes back truncated and the JSON fails to parse — the log says so and asks for a
  higher value.
* `timeout`: seconds before giving up on a call.
* `max_retries`: client-side attempts before returning an error.
* `concurrency`: how many years are processed in parallel within a report, via
  LangGraph's `max_concurrency`. Without it the fan-out would open all ten at once and
  run into the provider's rate limit. The effective ceiling is ten, the size of the grid.
* `image_format`: `jpeg` or `png` for the images sent. JPEG shrinks far more; PNG
  preserves fine table text.

## Output

Where `extract.py` writes its results. A single folder, with the file names inside it
fixed in the code.

* `output_dir`: receives `llm_extractions.csv`, the row-by-row extraction to be compared
  against the gold standard, and one `run_config_<run_id>.json` per execution holding
  the full config that produced those results. `--dry-run` writes into a `dry_run/`
  subdirectory so it never mixes with a real run.

## Qdrant

Connection and collection setup, used by both `ingest.py` and `extract.py` through
`store_from_config()`. The last four fields exist because ColQwen is multi-vector: there
are hundreds of vectors per page, and Qdrant's defaults would hold all of them in the
RAM the model needs. The distance metric is not configurable — MaxSim over ColQwen
assumes cosine.

* `url`: server address. `docker-compose.qdrant.yml` at the repository root brings up a
  local one.
* `collection_name`: a single collection for every report, partitioned by `doc_id` with
  a tenant index. One collection per document would blow past Qdrant's practical
  collection limit at 105 reports already.
* `prefer_grpc`: gRPC avoids the 32 MB JSON payload cap, which a multi-vector batch
  exceeds easily.
* `timeout`: seconds for client operations. Upserting a whole report is slow, hence the
  high value.
* `datatype`: storage precision for the vectors. `float16` loses nothing, since ColQwen
  already emits bf16.
* `memory`: `cold` keeps the vectors mmapped, with no preload, so Qdrant's RAM scales
  with the number of pages rather than the volume of vectors.
* `on_disk_payload`: keeps point payloads on disk instead of in RAM.
* `hnsw_m`: `0` disables the HNSW index — search becomes an exact full scan within the
  document, which is cheap because the tenant filter already narrows the universe to one
  report, and no graph is held in RAM. Side effect: `indexed_vectors_count` stays at 0
  and the collection may sit at `yellow` status forever; do not wait for `green`.
