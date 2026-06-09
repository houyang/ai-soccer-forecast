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

 check:
	make format
	make lint
	make typecheck
	make test
