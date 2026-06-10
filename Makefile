# Development targets. Run inside the project's environment
# (conda activate recension) or prefix with `conda run -n recension`.

.PHONY: test lint type check examples docs serve-docs demo build clean

test:
	pytest

lint:
	ruff check .

type:
	mypy

check: lint type test

examples:
	python examples/build_site.py

# `docs`/`serve-docs` need no `examples` prerequisite: the mkdocs hook
# (mkdocs_hooks.py) regenerates the example pages before every build.
docs:
	mkdocs build --strict

serve-docs:
	mkdocs serve

# Live demo: watch the optimizer run in the browser (needs the webapp extra:
# pip install "recension[webapp]"). Serves http://127.0.0.1:5001 (override with
# RECENSION_DEMO_PORT; 5000 is avoided because macOS AirPlay binds it).
demo:
	python examples/webapp/app.py

build:
	python -m build
	twine check dist/*

clean:
	rm -rf dist build site docs/examples/generated examples/output \
		.pytest_cache .mypy_cache .ruff_cache
