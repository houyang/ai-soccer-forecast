.PHONY: format lint typecheck test coverage check

format:
	ruff format .

lint:
	ruff check .

typecheck:
	mypy src tests

test:
	pytest

coverage:
	pytest --cov=soccer --cov-report=term-missing

check: format lint typecheck test
