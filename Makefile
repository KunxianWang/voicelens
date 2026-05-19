.PHONY: help up down init-db seed ingest-sample \
        ingest-amazon-fixture ingest-amazon-sample db-stats \
        scan-amazon-brands scan-amazon-brand-reviews generate-brand-allowlist profile-amazon-dataset \
        build-mvp-subset ingest-mvp-subset \
        seed-aspects absa-smoke absa-stats sample-absa-holdout \
        absa-llm-smoke absa-llm-1k absa-llm-holdout-50 export-absa-predictions evaluate-absa \
        validate-absa-labels evaluate-absa-50 export-absa-errors \
        embed-mock-smoke embed-v2-1k retrieval-smoke qdrant-stats \
        test lint install

PYTHON ?= python
ABSA_LLM_PROVIDER ?= anthropic
ABSA_MODEL ?= claude-opus-4.6
ABSA_MAX_COST_USD ?= 5
ABSA_ASPECT_VERSION ?= v2
QDRANT_COLLECTION ?= reviews_v2
EMBEDDING_PROVIDER ?= local
EMBEDDING_MODEL ?= BAAI/bge-small-en-v1.5

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
	@echo "  seed-aspects             seed aspect_ontology (default v2; pass --version v1 or --all directly to the script)"
	@echo "  absa-smoke               run absa_flow with MockABSAProvider on first 500 MVP-subset reviews"
	@echo "  absa-llm-smoke           run real LLM ABSA on 100 MVP-subset reviews"
	@echo "  absa-llm-1k              run real LLM ABSA on 1000 MVP-subset reviews"
	@echo "  absa-llm-holdout-50      run real LLM ABSA on review_ids listed in absa_holdout_labeled.jsonl"
	@echo "  absa-stats               print aspect_mention coverage / distribution / verbatim rate"
	@echo "  sample-absa-holdout      write data/labeling/absa_holdout_seed.jsonl for manual labeling"
	@echo "  export-absa-predictions  write data/labeling/absa_holdout_predictions.jsonl"
	@echo "  evaluate-absa            score labeled holdout against predictions"
	@echo "  validate-absa-labels     schema-check data/labeling/absa_holdout_labeled.jsonl"
	@echo "  evaluate-absa-50         partial-eval: only first 50 labeled rows"
	@echo "  export-absa-errors       export per-(review,aspect) error CSV for human review"
	@echo "  embed-mock-smoke         tiny mock-embedding indexing run (no network, in-memory Qdrant)"
	@echo "  embed-v2-1k              embed up to 1000 v2 ABSA-processed reviews into Qdrant"
	@echo "  retrieval-smoke          run a sample query against the Qdrant collection"
	@echo "  qdrant-stats             print collection size + aspect/sentiment distributions"
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
	$(PYTHON) scripts/seed_aspect_ontology.py --all

absa-smoke:
	$(PYTHON) -m voicelens.pipeline.flows.absa_flow \
	  --source amazon_reviews_2023_mvp_subset \
	  --limit 500 \
	  --provider mock

absa-llm-smoke:
	$(PYTHON) -m voicelens.pipeline.flows.absa_flow \
	  --source amazon_reviews_2023_mvp_subset \
	  --limit 100 \
	  --provider "$(ABSA_LLM_PROVIDER)" \
	  --model "$(ABSA_MODEL)" \
	  --aspect-version "$(ABSA_ASPECT_VERSION)" \
	  --llm-max-retries 2 \
	  --llm-retry-base-seconds 1.0 \
	  --min-processed-for-rate-guardrail 50 \
	  --max-invalid-rate 0.10 \
	  --max-fail-rate 0.05 \
	  --max-cost-usd 1

absa-llm-1k:
	$(PYTHON) -m voicelens.pipeline.flows.absa_flow \
	  --source amazon_reviews_2023_mvp_subset \
	  --limit 1000 \
	  --provider "$(ABSA_LLM_PROVIDER)" \
	  --model "$(ABSA_MODEL)" \
	  --aspect-version "$(ABSA_ASPECT_VERSION)" \
	  --llm-max-retries 2 \
	  --llm-retry-base-seconds 1.0 \
	  --min-processed-for-rate-guardrail 50 \
	  --max-invalid-rate 0.10 \
	  --max-fail-rate 0.05 \
	  --max-cost-usd "$(ABSA_MAX_COST_USD)"

absa-llm-holdout-50:
	$(PYTHON) -m voicelens.pipeline.flows.absa_flow \
	  --review-ids-file data/labeling/absa_holdout_labeled.jsonl \
	  --provider "$(ABSA_LLM_PROVIDER)" \
	  --model "$(ABSA_MODEL)" \
	  --aspect-version "$(ABSA_ASPECT_VERSION)" \
	  --llm-max-retries 2 \
	  --llm-retry-base-seconds 1.0 \
	  --min-processed-for-rate-guardrail 20 \
	  --max-invalid-rate 0.05 \
	  --max-fail-rate 0.10 \
	  --max-cost-usd 1

absa-stats:
	$(PYTHON) scripts/absa_stats.py

sample-absa-holdout:
	$(PYTHON) scripts/sample_absa_holdout.py --source amazon_reviews_2023_mvp_subset

export-absa-predictions:
	$(PYTHON) scripts/export_absa_predictions_for_labeling.py --model "$(ABSA_MODEL)" --aspect-version "$(ABSA_ASPECT_VERSION)"

embed-mock-smoke:
	$(PYTHON) -m voicelens.pipeline.flows.embed_flow \
	  --aspect-version "$(ABSA_ASPECT_VERSION)" \
	  --provider "$(ABSA_LLM_PROVIDER)" \
	  --model "$(ABSA_MODEL)" \
	  --embedding-provider mock \
	  --collection reviews_v2_mock_smoke \
	  --qdrant-url :memory: \
	  --limit 10

embed-v2-1k:
	$(PYTHON) -m voicelens.pipeline.flows.embed_flow \
	  --aspect-version "$(ABSA_ASPECT_VERSION)" \
	  --provider "$(ABSA_LLM_PROVIDER)" \
	  --model "$(ABSA_MODEL)" \
	  --embedding-provider "$(EMBEDDING_PROVIDER)" \
	  --embedding-model "$(EMBEDDING_MODEL)" \
	  --collection "$(QDRANT_COLLECTION)" \
	  --limit 1000

qdrant-stats:
	$(PYTHON) scripts/qdrant_stats.py --collection "$(QDRANT_COLLECTION)"

retrieval-smoke:
	$(PYTHON) scripts/retrieval_smoke.py \
	  --collection "$(QDRANT_COLLECTION)" \
	  --embedding-provider "$(EMBEDDING_PROVIDER)" \
	  --embedding-model "$(EMBEDDING_MODEL)" \
	  --query "product stopped working after a week" --limit 5

evaluate-absa:
	$(PYTHON) scripts/evaluate_absa.py

validate-absa-labels:
	$(PYTHON) scripts/validate_absa_labels.py

evaluate-absa-50:
	$(PYTHON) scripts/evaluate_absa.py --max-rows 50 --require-min-labeled 20

export-absa-errors:
	$(PYTHON) scripts/export_absa_eval_errors.py

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check voicelens scripts
