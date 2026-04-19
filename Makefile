.PHONY: install install-dev lint format test test-unit test-integration test-e2e run docker-up docker-down models clean

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt
	pre-commit install

lint:
	ruff check .
	black --check .
	mypy src config

format:
	ruff check --fix .
	black .

test: test-unit

test-unit:
	pytest tests/unit -m unit --cov=src --cov-report=term-missing

test-integration:
	pytest tests/integration -m integration

test-e2e:
	pytest tests/e2e -m e2e

run:
	streamlit run run.py

models:
	bash scripts/pull_models.sh

docker-up:
	docker compose up --build

docker-down:
	docker compose down

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
