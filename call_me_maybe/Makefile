install:
	uv sync

run:
	uv run python -m src $(ARGS)

debug:
	uv run python -m pdb src/__main__.py

clean:
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	find . -name ".mypy_cache" -type d -prune -exec rm -rf {} +

lint:
	flake8 . --exclude=.venv,__pycache__,.mypy_cache,.pytest_cache,llm_sdk
	mypy . \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs \
		--exclude '(\.venv|__pycache__|\.mypy_cache|\.pytest_cache|llm_sdk)'
