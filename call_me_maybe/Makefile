install:
	mkdir -p /goinfre/fchaput/.uv_cache
	UV_CACHE_DIR=/goinfre/fchaput/.uv_cache uv sync
	uv sync
	# rm -rf ~/.cache/uv to empty cache if needed

run:
	HF_HOME=/goinfre/$(USER)/huggingface \
	TRANSFORMERS_CACHE=/goinfre/$(USER)/huggingface \
	HUGGINGFACE_HUB_CACHE=/goinfre/$(USER)/huggingface \
	uv run python -m src "$(ARGS)"

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

batch:
	HF_HOME=/goinfre/$(USER)/huggingface \
	TRANSFORMERS_CACHE=/goinfre/$(USER)/huggingface \
	HUGGINGFACE_HUB_CACHE=/goinfre/$(USER)/huggingface \
	uv run python -m src.batch_runner

test:
	uv run python -m src.batch_runner

.PHONY: install run debug clean lint batch
