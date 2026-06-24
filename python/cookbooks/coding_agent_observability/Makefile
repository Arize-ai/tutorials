install:
	pip install ruff mypy pytest

check:
	ruff check .
	mypy store
	pytest -q

.PHONY: install check
