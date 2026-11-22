.DEFAULT_GOAL := help
UV ?= uv
PROFILE ?= offline

help: ## Show targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

sync: ## Install dependencies (incl. dev) from uv.lock
	$(UV) sync --frozen

fmt: ## Format code
	$(UV) run ruff format src tests
	$(UV) run ruff check --fix src tests

lint: ## Ruff lint + format check
	$(UV) run ruff check src tests
	$(UV) run ruff format --check src tests

type: ## mypy --strict
	$(UV) run mypy

test: ## All tests
	$(UV) run pytest -q

test-unit: ## Unit tests only
	$(UV) run pytest -q tests/unit

test-int: ## Integration tests (embedded stores, offline providers)
	$(UV) run pytest -q -m integration

check: lint type test ## Everything CI runs

index: ## Index all configured repos (PROFILE=offline|cloud|local)
	CODE_RAG_PROFILE=$(PROFILE) $(UV) run code-rag index

eval: ## Full benchmark on eval/questions.jsonl
	CODE_RAG_PROFILE=$(PROFILE) $(UV) run code-rag eval run --questions eval/questions.jsonl --out eval/reports/$(PROFILE).json

eval-fixture: ## Index the fixture corpus and run the curated benchmark (no API keys needed)
	$(UV) run python scripts/fixture_eval.py --profile $(PROFILE)

demo: ## End-to-end demo through the HTTP API (index, ask JSON/SSE, webhook re-index, guardrail)
	$(UV) run python scripts/demo.py --profile $(PROFILE)

up: ## docker compose up (api + worker + qdrant + redis)
	docker compose up -d --build

down: ## docker compose down
	docker compose down

logs: ## Tail service logs
	docker compose logs -f api worker

docker-build: ## Build the runtime image
	docker build -t code-rag:latest .

.PHONY: help sync fmt lint type test test-unit test-int check index eval eval-fixture demo up down logs docker-build
