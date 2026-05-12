install:
	uv sync

run:
	uv run python -m src $(ARGS)

debug:
	uv run python -m pdb -m src

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +

lint:
	flake8 .
	mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs