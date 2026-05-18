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

The reviews file itself **does not contain a `brand` field**. To filter by brand (Anker, Soundcore, Bose, JBL, UGREEN, RAVPower, Aukey) we look up brand from the metadata file by ASIN. If you skip the metadata file, the adapter falls back to any inline `brand` / `store` field that may be present on the review row, and otherwise drops the row.

## Other datasets

The same folder will be reused for later milestones (Trustpilot dumps, Reddit exports, etc.). Each subfolder will have its own `README` snippet here when it lands.
