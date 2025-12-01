#!/bin/bash

# Run all quality checks and tests

set -e

echo "======================================"
echo "Running Code Quality Checks"
echo "======================================"

echo ""
echo "1. Checking import order with isort..."
uv run isort --check-only backend/ main.py

echo ""
echo "2. Checking code formatting with black..."
uv run black --check backend/ main.py

echo ""
echo "3. Running flake8..."
uv run flake8 backend/ main.py

echo ""
echo "4. Running tests..."
cd backend && uv run pytest

echo ""
echo "======================================"
echo "All quality checks and tests passed!"
echo "======================================"
