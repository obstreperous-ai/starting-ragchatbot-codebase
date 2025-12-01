"""
Tests for CourseSearchTool.execute() method

These tests verify:
1. Successful search execution with various parameters
2. Error handling for empty results
3. Error handling for missing courses
4. Result formatting with different metadata
5. Source tracking functionality
"""

import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

# Add backend directory to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchToolExecute:
    """Test suite for CourseSearchTool.execute() method"""

    def test_execute_with_successful_results(
        self, populated_mock_vector_store, sample_search_results
    ):
        """Test execute returns formatted results when search succeeds"""
        # Arrange
        tool = CourseSearchTool(populated_mock_vector_store)
        populated_mock_vector_store.search.return_value = sample_search_results

        # Act
        result = tool.execute(query="computer use")

        # Assert
        assert result is not None
        assert isinstance(result, str)
        assert "Building Towards Computer Use with Anthropic" in result
        assert "Welcome to Building Toward Computer Use" in result
        assert "Lesson 0" in result
        populated_mock_vector_store.search.assert_called_once_with(
            query="computer use", course_name=None, lesson_number=None
        )

    def test_execute_with_course_filter(
        self, populated_mock_vector_store, sample_search_results
    ):
        """Test execute passes course_name filter to vector store"""
        # Arrange
        tool = CourseSearchTool(populated_mock_vector_store)
        populated_mock_vector_store.search.return_value = sample_search_results

        # Act
        result = tool.execute(query="tool use", course_name="Computer Use")

        # Assert
        assert result is not None
        populated_mock_vector_store.search.assert_called_once_with(
            query="tool use", course_name="Computer Use", lesson_number=None
        )

    def test_execute_with_lesson_filter(
        self, populated_mock_vector_store, sample_search_results
    ):
        """Test execute passes lesson_number filter to vector store"""
        # Arrange
        tool = CourseSearchTool(populated_mock_vector_store)
        populated_mock_vector_store.search.return_value = sample_search_results

        # Act
        result = tool.execute(query="introduction", lesson_number=0)

        # Assert
        assert result is not None
        populated_mock_vector_store.search.assert_called_once_with(
            query="introduction", course_name=None, lesson_number=0
        )

    def test_execute_with_both_filters(
        self, populated_mock_vector_store, sample_search_results
    ):
        """Test execute with both course and lesson filters"""
        # Arrange
        tool = CourseSearchTool(populated_mock_vector_store)
        populated_mock_vector_store.search.return_value = sample_search_results

        # Act
        result = tool.execute(
            query="tool basics", course_name="Computer Use", lesson_number=1
        )

        # Assert
        assert result is not None
        populated_mock_vector_store.search.assert_called_once_with(
            query="tool basics", course_name="Computer Use", lesson_number=1
        )

    def test_execute_with_empty_results(self, mock_vector_store, empty_search_results):
        """Test execute returns appropriate message when no results found"""
        # Arrange
        tool = CourseSearchTool(mock_vector_store)
        mock_vector_store.search.return_value = empty_search_results

        # Act
        result = tool.execute(query="nonexistent topic")

        # Assert
        assert result is not None
        assert "No relevant content found" in result

    def test_execute_with_empty_results_and_course_filter(
        self, mock_vector_store, empty_search_results
    ):
        """Test execute includes course name in 'no results' message"""
        # Arrange
        tool = CourseSearchTool(mock_vector_store)
        mock_vector_store.search.return_value = empty_search_results

        # Act
        result = tool.execute(query="test", course_name="Nonexistent Course")

        # Assert
        assert "No relevant content found" in result
        assert "Nonexistent Course" in result

    def test_execute_with_empty_results_and_lesson_filter(
        self, mock_vector_store, empty_search_results
    ):
        """Test execute includes lesson number in 'no results' message"""
        # Arrange
        tool = CourseSearchTool(mock_vector_store)
        mock_vector_store.search.return_value = empty_search_results

        # Act
        result = tool.execute(query="test", lesson_number=99)

        # Assert
        assert "No relevant content found" in result
        assert "lesson 99" in result

    def test_execute_with_search_error(self, mock_vector_store, error_search_results):
        """Test execute returns error message when search fails"""
        # Arrange
        tool = CourseSearchTool(mock_vector_store)
        mock_vector_store.search.return_value = error_search_results

        # Act
        result = tool.execute(query="test", course_name="Bad Course")

        # Assert
        assert result is not None
        assert "No course found matching" in result or "error" in result.lower()

    def test_format_results_creates_proper_headers(
        self, populated_mock_vector_store, sample_search_results
    ):
        """Test that formatted results include course and lesson headers"""
        # Arrange
        tool = CourseSearchTool(populated_mock_vector_store)
        populated_mock_vector_store.search.return_value = sample_search_results

        # Act
        result = tool.execute(query="test")

        # Assert
        assert "[Building Towards Computer Use with Anthropic - Lesson 0]" in result
        assert "[Building Towards Computer Use with Anthropic - Lesson 1]" in result

    def test_source_tracking(self, populated_mock_vector_store, sample_search_results):
        """Test that execute tracks sources correctly"""
        # Arrange
        tool = CourseSearchTool(populated_mock_vector_store)
        populated_mock_vector_store.search.return_value = sample_search_results

        # Act
        result = tool.execute(query="test")

        # Assert
        assert len(tool.last_sources) == 2
        assert (
            tool.last_sources[0]["text"]
            == "Building Towards Computer Use with Anthropic - Lesson 0"
        )
        assert tool.last_sources[0]["link"] == "https://example.com/lesson/0"
        assert (
            tool.last_sources[1]["text"]
            == "Building Towards Computer Use with Anthropic - Lesson 1"
        )
        assert tool.last_sources[1]["link"] == "https://example.com/lesson/1"

    def test_source_tracking_resets_on_new_search(
        self, populated_mock_vector_store, sample_search_results
    ):
        """Test that sources are reset for each new search"""
        # Arrange
        tool = CourseSearchTool(populated_mock_vector_store)
        populated_mock_vector_store.search.return_value = sample_search_results

        # Act - First search
        tool.execute(query="first search")
        first_sources = tool.last_sources.copy()

        # Act - Second search with different results
        single_result = SearchResults(
            documents=["Single result"],
            metadata=[
                {
                    "course_title": "Test Course",
                    "lesson_number": 5,
                    "lesson_link": "https://example.com/lesson/5",
                }
            ],
            distances=[0.1],
        )
        populated_mock_vector_store.search.return_value = single_result
        tool.execute(query="second search")

        # Assert
        assert len(first_sources) == 2
        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]["text"] == "Test Course - Lesson 5"

    def test_get_tool_definition(self, mock_vector_store):
        """Test that get_tool_definition returns proper schema"""
        # Arrange
        tool = CourseSearchTool(mock_vector_store)

        # Act
        definition = tool.get_tool_definition()

        # Assert
        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["properties"]["query"]["type"] == "string"
        assert "query" in definition["input_schema"]["required"]


class TestToolManager:
    """Test suite for ToolManager"""

    def test_register_tool(self, mock_vector_store):
        """Test tool registration"""
        # Arrange
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)

        # Act
        manager.register_tool(tool)

        # Assert
        assert "search_course_content" in manager.tools

    def test_get_tool_definitions(self, mock_vector_store):
        """Test getting all tool definitions"""
        # Arrange
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        # Act
        definitions = manager.get_tool_definitions()

        # Assert
        assert len(definitions) == 1
        assert definitions[0]["name"] == "search_course_content"

    def test_execute_tool(self, populated_mock_vector_store, sample_search_results):
        """Test executing a tool by name"""
        # Arrange
        manager = ToolManager()
        tool = CourseSearchTool(populated_mock_vector_store)
        manager.register_tool(tool)
        populated_mock_vector_store.search.return_value = sample_search_results

        # Act
        result = manager.execute_tool("search_course_content", query="test")

        # Assert
        assert result is not None
        assert isinstance(result, str)

    def test_execute_nonexistent_tool(self, mock_vector_store):
        """Test executing a tool that doesn't exist"""
        # Arrange
        manager = ToolManager()

        # Act
        result = manager.execute_tool("nonexistent_tool", query="test")

        # Assert
        assert "not found" in result.lower()

    def test_get_last_sources(self, populated_mock_vector_store, sample_search_results):
        """Test retrieving sources from last search"""
        # Arrange
        manager = ToolManager()
        tool = CourseSearchTool(populated_mock_vector_store)
        manager.register_tool(tool)
        populated_mock_vector_store.search.return_value = sample_search_results

        # Act
        manager.execute_tool("search_course_content", query="test")
        sources = manager.get_last_sources()

        # Assert
        assert len(sources) == 2
        assert (
            sources[0]["text"]
            == "Building Towards Computer Use with Anthropic - Lesson 0"
        )

    def test_reset_sources(self, populated_mock_vector_store, sample_search_results):
        """Test resetting sources"""
        # Arrange
        manager = ToolManager()
        tool = CourseSearchTool(populated_mock_vector_store)
        manager.register_tool(tool)
        populated_mock_vector_store.search.return_value = sample_search_results

        # Act
        manager.execute_tool("search_course_content", query="test")
        manager.reset_sources()
        sources = manager.get_last_sources()

        # Assert
        assert len(sources) == 0
