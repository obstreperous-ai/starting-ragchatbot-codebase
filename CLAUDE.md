# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Course Materials RAG (Retrieval-Augmented Generation) chatbot - a full-stack app that lets users query course content using semantic search and Claude AI responses.

## Commands

```bash
# Run the application (starts at http://localhost:8000)
./run.sh

# Or manually
cd backend && uv run uvicorn app:app --reload --port 8000

# Install dependencies
uv sync

# Code Quality Tools
./scripts/format.sh       # Auto-format code with black and isort
./scripts/lint.sh         # Run linting checks (isort, black, flake8)
./scripts/quality.sh      # Run all quality checks + tests

# Manual quality commands
uv run black backend/ main.py           # Format code
uv run isort backend/ main.py           # Sort imports
uv run flake8 backend/ main.py          # Lint code
```

## Architecture

### Backend Flow
1. **FastAPI** (`backend/app.py`) - Entry point with `/api/query` and `/api/courses` endpoints
2. **RAGSystem** (`backend/rag_system.py`) - Orchestrates all components
3. **AIGenerator** (`backend/ai_generator.py`) - Calls Claude API with tool definitions
4. **ToolManager/CourseSearchTool** (`backend/search_tools.py`) - Executes semantic searches when Claude requests
5. **VectorStore** (`backend/vector_store.py`) - ChromaDB wrapper for embeddings and search
6. **SessionManager** (`backend/session_manager.py`) - Maintains conversation history per session

### Document Processing
- `DocumentProcessor` (`backend/document_processor.py`) parses course docs from `docs/` folder
- Expected format: metadata header (title, link, instructor) followed by `Lesson N:` markers
- Content chunked at 800 chars with 100 char overlap, preserving sentence boundaries

### Query Processing
1. User query → FastAPI → RAGSystem
2. RAGSystem calls AIGenerator with tool definitions
3. Claude decides whether to call `search_course_content` tool
4. If tool called: VectorStore performs semantic search → results returned to Claude
5. Claude synthesizes final answer from search results
6. Response + sources returned to frontend

### Frontend
- Vanilla HTML/CSS/JS in `frontend/`
- Uses Marked.js for markdown rendering
- Maintains session ID for multi-turn conversations

## Configuration

Key settings in `backend/config.py`:
- `ANTHROPIC_MODEL`: claude-sonnet-4-20250514
- `EMBEDDING_MODEL`: all-MiniLM-L6-v2
- `CHUNK_SIZE`: 800 chars
- `CHUNK_OVERLAP`: 100 chars
- `MAX_RESULTS`: 5 search results
- `MAX_HISTORY`: 2 conversation exchanges

## Environment

Requires `ANTHROPIC_API_KEY` in `.env` file (see `.env.example`).
- Always use UV to run the server do not use PIP
- Make sure to use uv for ny dependencies
- Use uv to run python

## Code Quality

The project uses the following code quality tools:

- **Black** (v24+): Code formatter with 88 character line length
- **isort** (v5.13+): Import statement sorter, configured to work with black
- **flake8** (v7+): Linter for style guide enforcement

### Configuration

- `pyproject.toml`: Contains black and isort configuration
- `.flake8`: Contains flake8 configuration (88 char line length, compatible with black)

### Running Quality Checks

Before committing code, ensure all quality checks pass:

```bash
./scripts/format.sh    # Auto-format your code
./scripts/lint.sh      # Verify formatting and style
./scripts/quality.sh   # Run full quality suite including tests
```

### Development Workflow

1. Make your code changes
2. Run `./scripts/format.sh` to auto-format
3. Run `./scripts/lint.sh` to verify style compliance
4. Run tests with `cd backend && uv run pytest`
5. Or run everything with `./scripts/quality.sh`