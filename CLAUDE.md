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