# Data Reference

The gold standard used in this work is the one published by [Beck et al.](https://zenodo.org/records/14356664). We do not redistribute it, nor any derived variation of it, nor the PDF reports themselves.

`gold_coverage.json` compares the full gold standard with the subset used in our experiments — the entries whose PDF reports we were able to obtain.

`reports_guide.csv` lists exactly which reports make up that subset, one row per report with its company and source link.

> To replicate the experiments, download the gold standard from the link above and the PDFs listed in `reports_guide.csv`, then filter the gold standard down to those reports. Place the resulting CSV in `data/reference/available_gold_standard.csv` and the PDFs in `data/reports/`.
