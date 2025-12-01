# Proposed Fixes for RAG Chatbot

## Problem Summary
The RAG chatbot returns 'query failed' for content-related questions because:
1. **Vector store is empty** (0 courses loaded)
2. **.env file is missing** (no API key configured)
3. **Silent failures** in document loading process

## Fix Priority Levels
- 🔴 **CRITICAL**: System won't work without this
- 🟡 **HIGH**: Significantly improves reliability/debugging
- 🟢 **MEDIUM**: Nice to have, improves UX

---

## Fix #1: Create .env File 🔴 CRITICAL

### Problem
API key not configured, causing Anthropic API calls to fail.

### Solution
```bash
# Create .env file from example
cp .env.example .env

# Edit .env and add your actual API key
# ANTHROPIC_API_KEY=sk-ant-...
```

### Verification
```python
from config import config
assert len(config.ANTHROPIC_API_KEY) > 0, "API key must be set"
```

---

## Fix #2: Load Course Documents into Vector Store 🔴 CRITICAL

### Problem
Vector store is empty - no documents have been indexed.

### Root Cause
The startup event in `app.py` tries to load from `../docs` but:
- Path might be relative to wrong directory
- Errors might be silently caught
- Process might not run at all

### Solution A: Manual Document Loading (Immediate Fix)

Create a script to manually load documents:

```python
# backend/load_documents.py
import os
from config import config
from rag_system import RAGSystem

def load_all_documents():
    """Load all course documents into the vector store"""
    print("Initializing RAG system...")
    rag = RAGSystem(config)

    # Get absolute path to docs folder
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    docs_path = os.path.join(os.path.dirname(backend_dir), "docs")

    print(f"Looking for documents in: {docs_path}")

    if not os.path.exists(docs_path):
        print(f"ERROR: docs folder not found at {docs_path}")
        return

    # List available files
    files = [f for f in os.listdir(docs_path) if f.endswith('.txt')]
    print(f"Found {len(files)} text files: {files}")

    # Clear existing data and reload
    print("Clearing existing data...")
    rag.vector_store.clear_all_data()

    # Load documents
    print("Loading documents...")
    courses, chunks = rag.add_course_folder(docs_path, clear_existing=False)

    print(f"✅ Loaded {courses} courses with {chunks} total chunks")
    print(f"Course titles: {rag.vector_store.get_existing_course_titles()}")

    # Verify data was loaded
    count = rag.vector_store.get_course_count()
    if count == 0:
        print("⚠️  WARNING: No courses loaded! Check document format.")
    else:
        print(f"✅ SUCCESS: {count} courses in vector store")

if __name__ == "__main__":
    load_all_documents()
```

Run it:
```bash
cd backend
uv run python load_documents.py
```

### Solution B: Fix Startup Path (Long-term Fix)

Update `backend/app.py`:

```python
# backend/app.py
@app.on_event("startup")
async def startup_event():
    """Load initial documents on startup"""
    import os

    # Get absolute path to docs folder
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    docs_path = os.path.join(os.path.dirname(backend_dir), "docs")

    print(f"Checking for documents in: {docs_path}")

    if not os.path.exists(docs_path):
        print(f"⚠️  WARNING: docs folder not found at {docs_path}")
        print(f"    Current working directory: {os.getcwd()}")
        return

    print("Loading initial documents...")
    try:
        courses, chunks = rag_system.add_course_folder(docs_path, clear_existing=False)

        if courses == 0:
            print("⚠️  WARNING: No new courses loaded")
            print(f"    Existing courses: {rag_system.vector_store.get_existing_course_titles()}")
        else:
            print(f"✅ Loaded {courses} new courses with {chunks} chunks")

        # Always report total state
        total = rag_system.vector_store.get_course_count()
        print(f"✅ Total courses in database: {total}")

    except Exception as e:
        print(f"❌ ERROR loading documents: {e}")
        import traceback
        traceback.print_exc()
```

### Verification
```python
from config import config
from vector_store import VectorStore

store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS)
print(f"Courses loaded: {store.get_course_count()}")
assert store.get_course_count() > 0, "No courses loaded!"
```

---

## Fix #3: Add Logging Infrastructure 🟡 HIGH

### Problem
Errors are silently caught, making debugging difficult.

### Solution

Create `backend/logging_config.py`:
```python
import logging
import sys

def setup_logging():
    """Configure logging for the application"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('rag_system.log')
        ]
    )

    # Set specific loggers
    logging.getLogger('chromadb').setLevel(logging.WARNING)
    logging.getLogger('anthropic').setLevel(logging.INFO)

    return logging.getLogger(__name__)

logger = setup_logging()
```

Update key files to use logging:

```python
# backend/rag_system.py
from logging_config import logger

class RAGSystem:
    def query(self, query: str, session_id: Optional[str] = None):
        logger.info(f"Processing query: {query[:100]}...")

        try:
            # ... existing code ...
            logger.info(f"Generated response with {len(sources)} sources")
            return response, sources
        except Exception as e:
            logger.error(f"Query failed: {e}", exc_info=True)
            raise
```

---

## Fix #4: Improve Error Messages 🟡 HIGH

### Problem
User sees generic "query failed" without details.

### Solution A: Better Exception Handling in app.py

```python
# backend/app.py
@app.post("/api/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Process a query and return response with sources"""
    try:
        # Create session if not provided
        session_id = request.session_id
        if not session_id:
            session_id = rag_system.session_manager.create_session()

        # Check if vector store has data
        course_count = rag_system.vector_store.get_course_count()
        if course_count == 0:
            return QueryResponse(
                answer="⚠️ No course materials have been loaded yet. Please contact the administrator to load course documents.",
                sources=[],
                session_id=session_id
            )

        # Process query using RAG system
        answer, sources = rag_system.query(request.query, session_id)

        # Convert source dicts to SourceItem objects
        source_items = [
            SourceItem(text=s.get("text", ""), link=s.get("link"))
            for s in sources
        ]

        return QueryResponse(
            answer=answer,
            sources=source_items,
            session_id=session_id
        )
    except Exception as e:
        # Log the full error
        import traceback
        print(f"Query error: {e}")
        traceback.print_exc()

        # Return user-friendly message
        error_msg = "An error occurred processing your query. "

        # Check for common issues
        if "ANTHROPIC_API_KEY" in str(e) or "api_key" in str(e).lower():
            error_msg += "API key not configured."
        elif "No course found" in str(e):
            error_msg += "Course not found in database."
        else:
            error_msg += f"Details: {str(e)}"

        raise HTTPException(status_code=500, detail=error_msg)
```

### Solution B: Add Health Check Endpoint

```python
# backend/app.py
@app.get("/api/health")
async def health_check():
    """Check system health and configuration"""
    from config import config

    health = {
        "status": "ok",
        "issues": [],
        "statistics": {}
    }

    # Check API key
    if not config.ANTHROPIC_API_KEY:
        health["status"] = "error"
        health["issues"].append("ANTHROPIC_API_KEY not configured")

    # Check vector store
    try:
        course_count = rag_system.vector_store.get_course_count()
        health["statistics"]["courses_loaded"] = course_count

        if course_count == 0:
            health["status"] = "warning"
            health["issues"].append("No courses loaded in vector store")
    except Exception as e:
        health["status"] = "error"
        health["issues"].append(f"Vector store error: {str(e)}")

    # Check docs folder
    import os
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    docs_path = os.path.join(os.path.dirname(backend_dir), "docs")

    if not os.path.exists(docs_path):
        health["issues"].append(f"docs folder not found at {docs_path}")
    else:
        doc_files = [f for f in os.listdir(docs_path) if f.endswith('.txt')]
        health["statistics"]["doc_files_available"] = len(doc_files)

    return health
```

Test it:
```bash
curl http://localhost:8000/api/health
```

---

## Fix #5: Add Database Initialization Check 🟢 MEDIUM

### Problem
No way to know if database needs initialization.

### Solution

Add to `backend/app.py`:

```python
@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    import os
    from logging_config import logger

    # Check database state
    course_count = rag_system.vector_store.get_course_count()
    logger.info(f"Startup: Vector store has {course_count} courses")

    # Only load documents if database is empty
    if course_count == 0:
        logger.warning("Vector store is empty - loading documents...")

        backend_dir = os.path.dirname(os.path.abspath(__file__))
        docs_path = os.path.join(os.path.dirname(backend_dir), "docs")

        if os.path.exists(docs_path):
            try:
                courses, chunks = rag_system.add_course_folder(docs_path, clear_existing=False)
                logger.info(f"✅ Loaded {courses} courses with {chunks} chunks")
            except Exception as e:
                logger.error(f"❌ Failed to load documents: {e}", exc_info=True)
        else:
            logger.error(f"❌ docs folder not found at {docs_path}")
    else:
        logger.info(f"✅ Vector store already populated with {course_count} courses")
```

---

## Implementation Checklist

### Immediate Actions (Do First)
- [ ] Create `.env` file with valid `ANTHROPIC_API_KEY`
- [ ] Run `load_documents.py` to populate vector store
- [ ] Verify with health check that courses are loaded
- [ ] Test a query: `curl -X POST http://localhost:8000/api/query -H "Content-Type: application/json" -d '{"query": "What is computer use?"}'`

### Short-term Improvements
- [ ] Add logging infrastructure
- [ ] Improve error messages in `app.py`
- [ ] Add health check endpoint
- [ ] Fix startup path to use absolute paths

### Long-term Enhancements
- [ ] Add database migration scripts
- [ ] Add admin endpoint to reload documents
- [ ] Add monitoring/alerting for empty database
- [ ] Add unit tests for document loading

---

## Testing After Fixes

### 1. Test Vector Store
```bash
cd backend
uv run python -c "
from config import config
from vector_store import VectorStore
store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS)
count = store.get_course_count()
print(f'Courses: {count}')
assert count > 0, 'No courses loaded!'
print('✅ Vector store OK')
"
```

### 2. Test RAG System
```bash
cd backend
uv run python -c "
from config import config
from rag_system import RAGSystem
rag = RAGSystem(config)
response, sources = rag.query('What is computer use?')
print(f'Response: {response[:100]}...')
print(f'Sources: {len(sources)}')
assert len(response) > 0, 'Empty response!'
print('✅ RAG system OK')
"
```

### 3. Test Full API
```bash
# Start server
./run.sh

# In another terminal:
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is computer use?"}' | jq .

# Should return actual course content, not an error
```

### 4. Run Test Suite
```bash
cd backend
uv run pytest tests/ -v
# Should see 68+ passing tests
```

---

## Expected Outcome

After implementing these fixes:

1. ✅ Vector store will have 4 courses loaded
2. ✅ Queries will return relevant course content
3. ✅ Sources will include lesson links
4. ✅ Error messages will be informative
5. ✅ Health check will show system status
6. ✅ Logs will help debugging

The chatbot will transform from returning "query failed" to providing helpful, accurate responses based on the indexed course materials.
