#!/bin/bash

# Run code quality checks

set -e

echo "Checking import order with isort..."
uv run isort --check-only backend/ main.py

echo "Checking code formatting with black..."
uv run black --check backend/ main.py

echo "Running flake8..."
uv run flake8 backend/ main.py

echo "All quality checks passed!"
