.PHONY: py-install py-lint py-format py-test rust-check

py-install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

py-lint:
	ruff check .

py-format:
	ruff format .

py-test:
	pytest

rust-check:
	cargo fmt
	cargo clippy