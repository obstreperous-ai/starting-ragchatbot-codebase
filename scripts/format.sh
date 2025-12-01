#!/bin/bash

# Format Python code with black and isort

set -e

echo "Running isort..."
uv run isort backend/ main.py

echo "Running black..."
uv run black backend/ main.py

echo "Code formatting complete!"
