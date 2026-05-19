# `data/` — Local datasets (not committed)

This folder holds **large user-supplied datasets**. It is excluded from git via `.gitignore` (only this README is committed).

## Amazon Reviews 2023

Source: McAuley Lab, UCSD — <https://amazon-reviews-2023.github.io/>

Place one or both of:

```
data/
  Electronics.jsonl              # the reviews file (5-core or full)
  meta_Electronics.jsonl         # optional metadata file (used to resolve brand by ASIN)
```

Gzipped variants are also supported:

```
data/
  Electronics.jsonl.gz
  meta_Electronics.jsonl.gz
```

Configure paths via `.env` (`AMAZON_REVIEWS_PATH`, `AMAZON_METADATA_PATH`).

## Why two files

The reviews file itself **does not contain a `brand` field**. First scan candidate brands with `BRAND_CANDIDATES` and generate `data/resolved_brand_allowlist.json`; then use the resolved brands as the runtime `BRAND_ALLOWLIST`. If you skip the metadata file, the ingest adapter falls back to any inline `brand` / `store` field that may be present on the review row, and otherwise drops the row.

Expected profiling outputs:

```
data/brand_inventory.csv
data/brand_review_counts.csv
data/resolved_brand_allowlist.json
data/dataset_profile.json
```

## Other datasets

The same folder will be reused for later milestones (Trustpilot dumps, Reddit exports, etc.). Each subfolder will have its own `README` snippet here when it lands.
