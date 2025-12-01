"""
Pytest configuration and shared fixtures for RAG system tests
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from typing import List, Dict, Any
import tempfile
import shutil

# Add backend directory to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from models import Course, Lesson, CourseChunk
from vector_store import SearchResults


@pytest.fixture
def sample_course():
    """Create a sample course for testing"""
    return Course(
        title="Building Towards Computer Use with Anthropic",
        course_link="https://www.deeplearning.ai/short-courses/building-toward-computer-use-with-anthropic/",
        instructor="Colt Steele",
        lessons=[
            Lesson(
                lesson_number=0,
                title="Introduction",
                lesson_link="https://learn.deeplearning.ai/courses/building-toward-computer-use-with-anthropic/lesson/a6k0z/introduction"
            ),
            Lesson(
                lesson_number=1,
                title="Tool Use Basics",
                lesson_link="https://learn.deeplearning.ai/courses/building-toward-computer-use-with-anthropic/lesson/b7l1a/tool-use-basics"
            )
        ]
    )


@pytest.fixture
def sample_course_chunks():
    """Create sample course chunks for testing"""
    return [
        CourseChunk(
            content="Welcome to Building Toward Computer Use with Anthropic. Built in partnership with Anthropic and taught by Colt Steele.",
            course_title="Building Towards Computer Use with Anthropic",
            lesson_number=0,
            chunk_index=0,
            lesson_link="https://learn.deeplearning.ai/courses/building-toward-computer-use-with-anthropic/lesson/a6k0z/introduction"
        ),
        CourseChunk(
            content="In this lesson, you'll learn about tool use and how Claude can interact with external tools.",
            course_title="Building Towards Computer Use with Anthropic",
            lesson_number=1,
            chunk_index=1,
            lesson_link="https://learn.deeplearning.ai/courses/building-toward-computer-use-with-anthropic/lesson/b7l1a/tool-use-basics"
        )
    ]


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store for testing"""
    mock_store = Mock()

    # Default search returns empty results
    mock_store.search.return_value = SearchResults(
        documents=[],
        metadata=[],
        distances=[],
        error=None
    )

    mock_store.get_existing_course_titles.return_value = []
    mock_store.get_course_count.return_value = 0

    return mock_store


@pytest.fixture
def populated_mock_vector_store(sample_course_chunks):
    """Create a mock vector store with sample data"""
    mock_store = Mock()

    # Simulate successful search with results
    mock_store.search.return_value = SearchResults(
        documents=[chunk.content for chunk in sample_course_chunks],
        metadata=[{
            'course_title': chunk.course_title,
            'lesson_number': chunk.lesson_number,
            'lesson_link': chunk.lesson_link
        } for chunk in sample_course_chunks],
        distances=[0.1, 0.2],
        error=None
    )

    mock_store.get_existing_course_titles.return_value = ["Building Towards Computer Use with Anthropic"]
    mock_store.get_course_count.return_value = 1

    return mock_store


@pytest.fixture
def mock_anthropic_client():
    """Create a mock Anthropic client for testing"""
    mock_client = Mock()

    # Default response without tool use
    mock_response = Mock()
    mock_response.stop_reason = "end_turn"
    mock_response.content = [Mock(text="This is a test response", type="text")]

    mock_client.messages.create.return_value = mock_response

    return mock_client


@pytest.fixture
def mock_anthropic_client_with_tool_use():
    """Create a mock Anthropic client that simulates tool calling"""
    mock_client = Mock()

    # First response: tool use
    tool_use_block = Mock()
    tool_use_block.type = "tool_use"
    tool_use_block.id = "tool_123"
    tool_use_block.name = "search_course_content"
    tool_use_block.input = {"query": "computer use"}

    tool_response = Mock()
    tool_response.stop_reason = "tool_use"
    tool_response.content = [tool_use_block]

    # Second response: final answer after tool execution
    final_response = Mock()
    final_response.stop_reason = "end_turn"
    final_response.content = [Mock(
        text="Based on the search results, computer use allows Claude to interact with computers.",
        type="text"
    )]

    # Configure mock to return different responses on subsequent calls
    mock_client.messages.create.side_effect = [tool_response, final_response]

    return mock_client


@pytest.fixture
def temp_chroma_path():
    """Create a temporary directory for ChromaDB testing"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup after test
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_config():
    """Create a mock config object for testing"""
    config = Mock()
    config.ANTHROPIC_API_KEY = "test-api-key"
    config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
    config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    config.CHUNK_SIZE = 800
    config.CHUNK_OVERLAP = 100
    config.MAX_RESULTS = 5
    config.MAX_HISTORY = 2
    config.CHROMA_PATH = "./test_chroma_db"
    return config


@pytest.fixture
def sample_search_results():
    """Create sample search results for testing"""
    return SearchResults(
        documents=[
            "Welcome to Building Toward Computer Use with Anthropic.",
            "Tool use allows Claude to interact with external systems."
        ],
        metadata=[
            {
                'course_title': "Building Towards Computer Use with Anthropic",
                'lesson_number': 0,
                'lesson_link': "https://example.com/lesson/0"
            },
            {
                'course_title': "Building Towards Computer Use with Anthropic",
                'lesson_number': 1,
                'lesson_link': "https://example.com/lesson/1"
            }
        ],
        distances=[0.1, 0.2]
    )


@pytest.fixture
def empty_search_results():
    """Create empty search results for testing"""
    return SearchResults(
        documents=[],
        metadata=[],
        distances=[]
    )


@pytest.fixture
def error_search_results():
    """Create error search results for testing"""
    return SearchResults(
        documents=[],
        metadata=[],
        distances=[],
        error="No course found matching 'NonExistent Course'"
    )


@pytest.fixture
def mock_rag_system():
    """Create a mock RAG system for API testing"""
    mock_rag = Mock()

    # Default successful query response
    mock_rag.query.return_value = (
        "This is a test answer about computer use.",
        [
            {
                "text": "Welcome to Building Toward Computer Use",
                "link": "https://example.com/lesson/0"
            }
        ]
    )

    # Default course analytics
    mock_rag.get_course_analytics.return_value = {
        "total_courses": 1,
        "course_titles": ["Building Towards Computer Use with Anthropic"]
    }

    # Mock session manager
    mock_rag.session_manager = Mock()
    mock_rag.session_manager.create_session.return_value = "test-session-123"

    return mock_rag


@pytest.fixture
def sample_query_request():
    """Create a sample query request payload"""
    return {
        "query": "What is computer use?",
        "session_id": "test-session-123"
    }


@pytest.fixture
def sample_query_request_no_session():
    """Create a sample query request without session ID"""
    return {
        "query": "What is computer use?"
    }


@pytest.fixture
def sample_query_response():
    """Create a sample query response"""
    return {
        "answer": "Computer use allows Claude to interact with computers.",
        "sources": [
            {
                "text": "Welcome to Building Toward Computer Use",
                "link": "https://example.com/lesson/0"
            }
        ],
        "session_id": "test-session-123"
    }


@pytest.fixture
def sample_course_stats():
    """Create sample course statistics"""
    return {
        "total_courses": 2,
        "course_titles": [
            "Building Towards Computer Use with Anthropic",
            "Introduction to LLMs"
        ]
    }
