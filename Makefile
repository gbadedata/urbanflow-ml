.PHONY: help install test lint typecheck check pipeline db serve image clean

help:
	@echo "install    install dependencies"
	@echo "check      lint, typecheck and test (what CI runs)"
	@echo "pipeline   ingest through features, from raw TLC data"
	@echo "db         start PostgreSQL and load zones and grid"
	@echo "serve      run the API against the local database"
	@echo "image      build the production container"

install:
	uv sync --all-groups

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy src

check: lint typecheck test

pipeline:
	uv run urbanflow ingest
	uv run urbanflow contract-check
	uv run urbanflow validate
	uv run urbanflow zone-report
	uv run urbanflow build-grid
	uv run urbanflow build-features

db:
	docker compose up -d postgres
	uv run urbanflow load-db

serve:
	uv run urbanflow serve --reload

image:
	docker build -f docker/Dockerfile -t urbanflow-ml:local .

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
