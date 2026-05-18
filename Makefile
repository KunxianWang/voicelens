.PHONY: help up down init-db seed ingest-sample \
        ingest-amazon-fixture ingest-amazon-sample db-stats \
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

db-stats:
	$(PYTHON) scripts/db_stats.py

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check voicelens scripts
