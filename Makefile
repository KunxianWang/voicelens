.PHONY: help up down init-db seed ingest-sample \
        ingest-amazon-fixture ingest-amazon-sample db-stats \
        scan-amazon-brands scan-amazon-brand-reviews generate-brand-allowlist profile-amazon-dataset \
        build-mvp-subset ingest-mvp-subset \
        seed-aspects absa-smoke absa-stats sample-absa-holdout \
        test lint install

PYTHON ?= python

help:
	@echo "Targets:"
	@echo "  install                  pip install -e .[dev]"
	@echo "  up                       start Postgres + Qdrant via docker compose"
	@echo "  down                     stop docker compose"
	@echo "  init-db                  create all tables in Postgres"
	@echo "  seed                     generate voicelens/data/sample_reviews.jsonl"
	@echo "  ingest-sample            run ingest_flow on synthetic sample"
	@echo "  ingest-amazon-fixture    run amazon_ingest_flow on the 20-row committed fixture"
	@echo "  ingest-amazon-sample     run amazon_ingest_flow on \$$AMAZON_REVIEWS_PATH (real subset)"
	@echo "  scan-amazon-brands       scan metadata for candidate brand availability"
	@echo "  scan-amazon-brand-reviews join reviews to metadata brand map and count reviews"
	@echo "  generate-brand-allowlist write data/resolved_brand_allowlist.json from scan outputs"
	@echo "  profile-amazon-dataset   write data/dataset_profile.json from metadata + reviews"
	@echo "  build-mvp-subset         build deterministic Amazon MVP subset JSONL.GZ"
	@echo "  ingest-mvp-subset        ingest data/amazon_mvp_reviews.jsonl.gz into Postgres"
	@echo "  seed-aspects             seed aspect_ontology v1 (idempotent)"
	@echo "  absa-smoke               run absa_flow with MockABSAProvider on first 500 MVP-subset reviews"
	@echo "  absa-stats               print aspect_mention coverage / distribution / verbatim rate"
	@echo "  sample-absa-holdout      write data/labeling/absa_holdout_seed.jsonl for manual labeling"
	@echo "  db-stats                 print review / brand / rating / DQ counts"
	@echo "  test                     run pytest (uses SQLite in-memory)"
	@echo "  lint                     ruff check"

install:
	$(PYTHON) -m pip install -e ".[dev]"

up:
	docker compose -f ops/docker/compose.yaml up -d

down:
	docker compose -f ops/docker/compose.yaml down

init-db:
	$(PYTHON) scripts/init_db.py

seed:
	$(PYTHON) scripts/seed_sample.py

ingest-sample:
	$(PYTHON) -m voicelens.pipeline.flows.ingest_flow

ingest-amazon-fixture:
	$(PYTHON) -m voicelens.pipeline.flows.amazon_ingest_flow \
	  --input voicelens/data/amazon_reviews_fixture.jsonl

ingest-amazon-sample:
	@if [ -z "$$AMAZON_REVIEWS_PATH" ] && [ ! -f data/Electronics.jsonl ] && [ ! -f data/Electronics.jsonl.gz ]; then \
	  echo "ERROR: AMAZON_REVIEWS_PATH is unset and no default file found."; \
	  echo "       Drop Amazon Reviews 2023 Electronics file at data/Electronics.jsonl(.gz)"; \
	  echo "       or set AMAZON_REVIEWS_PATH in .env. See data/README.md."; \
	  exit 1; \
	fi
	$(PYTHON) -m voicelens.pipeline.flows.amazon_ingest_flow

scan-amazon-brands:
	$(PYTHON) -m voicelens.ingest.brand_scan

scan-amazon-brand-reviews:
	$(PYTHON) -m voicelens.ingest.brand_review_scan $(if $(LIMIT),--limit $(LIMIT),--full-scan)

generate-brand-allowlist:
	$(PYTHON) scripts/generate_brand_allowlist.py

profile-amazon-dataset:
	$(PYTHON) scripts/profile_dataset.py --write $(if $(LIMIT),--limit $(LIMIT),--full-scan)

build-mvp-subset:
	$(PYTHON) scripts/build_mvp_subset.py

ingest-mvp-subset:
	$(PYTHON) -m voicelens.pipeline.flows.amazon_ingest_flow \
	  --input data/amazon_mvp_reviews.jsonl.gz \
	  --metadata data/meta_Electronics.jsonl.gz \
	  --brands-file data/resolved_brand_allowlist.json \
	  --no-limit \
	  --source amazon_reviews_2023_mvp_subset

db-stats:
	$(PYTHON) scripts/db_stats.py

seed-aspects:
	$(PYTHON) scripts/seed_aspect_ontology.py

absa-smoke:
	$(PYTHON) -m voicelens.pipeline.flows.absa_flow \
	  --source amazon_reviews_2023_mvp_subset \
	  --limit 500 \
	  --provider mock

absa-stats:
	$(PYTHON) scripts/absa_stats.py

sample-absa-holdout:
	$(PYTHON) scripts/sample_absa_holdout.py --source amazon_reviews_2023_mvp_subset

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check voicelens scripts
