format:
	python -m black . --config black.toml
prep-docs:
	typer fmengine/cli/main.py utils docs --name "fmengine" --output docs/source/cli.md --title "FMEngine CLI"