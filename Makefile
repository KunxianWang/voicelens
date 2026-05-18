.PHONY: help up down init-db seed ingest-sample test lint install

PYTHON ?= python

help:
	@echo "Targets:"
	@echo "  install        pip install -e .[dev]"
	@echo "  up             start Postgres + Qdrant via docker compose"
	@echo "  down           stop docker compose"
	@echo "  init-db        create all tables in Postgres"
	@echo "  seed           generate voicelens/data/sample_reviews.jsonl"
	@echo "  ingest-sample  run ingest_flow on sample data"
	@echo "  test           run pytest (uses SQLite in-memory)"
	@echo "  lint           ruff check"

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

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check voicelens scripts
