# Third-Party Notices

The code in this repository is released under the MIT License (see [`LICENSE`](LICENSE)).
The pipeline it implements, however, depends on external components — a benchmark
dataset, copyrighted PDF reports, pretrained model weights, a hosted API and a set of
Python packages — that carry their own licenses and terms. **None of these components
are redistributed here.** They are downloaded or accessed at run time, and complying
with their licenses is the responsibility of whoever runs the pipeline.

Some of these terms are more restrictive than MIT. In particular, the GIST gold
standard dataset and the Qwen2.5-VL-3B-Instruct base model both restrict use to
non-commercial purposes, and PyMuPDF is copyleft. Read the sections below before
using this pipeline in a commercial or redistributable setting.

## GIST Gold Standard Dataset (Beck et al., 2025)

- **License:** CC BY-NC-ND 1.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 1.0 Generic).
- **Status:** not included in this repository. It must be obtained by the user directly from Zenodo.
- **Version used:** v8, published 2026-02-19 — DOI [`10.5281/zenodo.18696096`](https://doi.org/10.5281/zenodo.18696096).
- **Concept DOI (all versions):** [`10.5281/zenodo.14035800`](https://doi.org/10.5281/zenodo.14035800).

Implications:

- Use is restricted to non-commercial purposes.
- No modified or combined version of the dataset may be redistributed. This includes
  filtered subsets, reshaped evaluation tables, and any file that merges gold standard
  values with other data.
- Accordingly, this repository publishes only the predictions produced by the pipeline
  itself, plus the scripts that rebuild the evaluation table locally from the dataset
  the user downloads.

Citation:

> Beck, J., Steinberg, A., Dimmelmeier, A., Domenech Burin, L., Kormanyos, E., Fehr, M., & Schierholz, M. (2025). Addressing data gaps in sustainability reporting: A benchmark dataset for greenhouse gas emission extraction. *Scientific Data*, 12, 1497. https://doi.org/10.1038/s41597-025-05664-8

Note: the authors' R code (https://github.com/soda-lmu/gist-data-descriptor) does not
carry a license file. No portion of it has been copied into this repository.

## Sustainability reports (PDF)

The PDF reports listed in [`data/reference/reports_guide.csv`](data/reference/reports_guide.csv)
are copyrighted works of the respective companies. They are **not redistributed** here.
The CSV contains only the report name, the company, and the public URL each document was
originally published at.

## ColQwen2.5-3b-multilingual-v1.0

- **Model id:** `Metric-AI/ColQwen2.5-3b-multilingual-v1.0` (configured in
  [`config/config.yml`](config/config.yml) as `extraction.col_extraction_model`).
- **LoRA adapters:** MIT.
- **Base model:** `Metric-AI/colqwen2.5-3b-base`, a ColQwen2.5 repackaging of
  `Qwen/Qwen2.5-VL-3B-Instruct`, and therefore covered by the **Qwen Research License
  Agreement** — use restricted to research / non-commercial purposes.
  https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/LICENSE
- **Weights are not included** in this repository. They are downloaded at run time by
  `transformers` / `colpali-engine`.

Note: the model card of the sibling T-Systems ColQwen2.5-3b-multilingual release states
Apache 2.0 for the backbone. That is incorrect for the 3B variant — Apache 2.0 applies
to Qwen2.5-VL-7B-Instruct, not to the 3B one. The license in Qwen's official repository
prevails.

Citation:

> Faysse, M., Sibille, H., Wu, T., Omrani, B., Viaud, G., Hudelot, C., & Colombo, P. (2024). ColPali: Efficient Document Retrieval with Vision Language Models. arXiv:2407.01449.

## OpenAI API

There is no software license to declare for the extraction model: it is consumed as a
hosted service through the `openai` Python client.

- **Model configured for extraction:** `gpt-5.6-luna` (see `vlm.model` in
  [`config/config.yml`](config/config.yml)).
- The retrieved report pages are sent as images to a third-party service and are
  therefore subject to OpenAI's terms of use and data policies.

## Python dependencies

Licenses as declared by the distributions installed in this project's environment.

| Package | License | Notes |
|---|---|---|
| accelerate | Apache-2.0 | |
| transformers | Apache-2.0 | installed from git (see `[tool.uv.sources]`) |
| openai | Apache-2.0 | |
| qdrant-client | Apache-2.0 | |
| colpali-engine | MIT | installed from git (see `[tool.uv.sources]`) |
| fast-plaid | MIT | |
| langgraph | MIT | |
| pydantic | MIT | |
| pyyaml | MIT | |
| pillow | MIT-CMU (HPND) | |
| torch | BSD-3-Clause | |
| jupyterlab | BSD-3-Clause | development dependency only |
| pymupdf | **AGPL-3.0** (or Artifex commercial license) | The only copyleft dependency. Used solely to render PDF pages as images. Distributing a binary or container with PyMuPDF bundled may subject the combined work to the AGPL. |
