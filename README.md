# Reading Sustainability Reports as Pages, Not Text: a visual retrieval and extraction architecture for GHG metrics

Extraction of GHG emissions (Scope 1, 2mb, 2lb, 3 × 2013–2022) from
sustainability reports in PDF using visual RAG: pages as images, ColQwen2.5 +
Qdrant for retrieval, and a VLM for reading.

## License

The code in this repository is released under the MIT License — see [`LICENSE`](LICENSE).

The pipeline depends on external components with more restrictive terms: the GIST gold
standard dataset (CC BY-NC-ND 1.0) and the Qwen2.5-VL-3B-Instruct base model behind
ColQwen2.5 (Qwen Research License) both restrict use to non-commercial purposes. Neither
the dataset nor the PDF reports are redistributed here — see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for the full list of components and
their licenses.
