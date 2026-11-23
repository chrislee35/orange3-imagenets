# Makefile for building Sphinx docs and Python package

.PHONY: all docs build clean test ruff ruff-fix black flake8 pylint lint release

# Build everything: docs + package
all: docs build

# Build Sphinx documentation
docs:
	sphinx-build -b html doc/ doc/_build/html

# Build the Python package (wheel and source)
build:
	python -m build

# Push package to pypi.org
release: build
	twine upload dist/*

# Clean build artifacts
clean:
	rm -rf build/ dist/ *.egg-info doc/_build

test:
	QT_QPA_PLATFORM=offscreen python -m pytest tests -q

ruff:
	ruff check orangecontrib tests

ruff-fix:
	ruff check --fix orangecontrib tests

black:
	black orangecontrib tests

flake8:
	flake8 orangecontrib tests

pylint:
	pylint orangecontrib tests

lint: ruff flake8 pylint
	black --check orangecontrib tests
