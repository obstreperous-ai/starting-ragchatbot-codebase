"""
Tests for VectorStore functionality

These tests verify:
1. Search functionality with various filters
2. Course name resolution
3. Filter building
4. Empty database scenarios
5. Data persistence
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile
import shutil

# Add backend directory to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from vector_store import VectorStore, SearchResults
from models import Course, Lesson, CourseChunk


class TestVectorStoreInitialization:
    """Test suite for VectorStore initialization"""

    def test_initialization_creates_collections(self, temp_chroma_path):
        """Test that VectorStore creates necessary collections"""
        # Act
        store = VectorStore(
            chroma_path=temp_chroma_path,
            embedding_model="all-MiniLM-L6-v2",
            max_results=5
        )

        # Assert
        assert store.course_catalog is not None
        assert store.course_content is not None
        assert store.max_results == 5

    def test_initialization_persists_data(self, temp_chroma_path, sample_course, sample_course_chunks):
        """Test that data persists across VectorStore instances"""
        # Arrange & Act - First instance: add data
        store1 = VectorStore(temp_chroma_path, "all-MiniLM-L6-v2", 5)
        store1.add_course_metadata(sample_course)
        store1.add_course_content(sample_course_chunks)

        # Act - Second instance: check data exists
        store2 = VectorStore(temp_chroma_path, "all-MiniLM-L6-v2", 5)
        course_count = store2.get_course_count()

        # Assert
        assert course_count == 1


class TestVectorStoreSearch:
    """Test suite for VectorStore search functionality"""

    def test_search_empty_database_returns_empty_results(self, temp_chroma_path):
        """Test searching empty database returns no results"""
        # Arrange
        store = VectorStore(temp_chroma_path, "all-MiniLM-L6-v2", 5)

        # Act
        results = store.search(query="test query")

        # Assert
        assert results.is_empty()
        assert len(results.documents) == 0
        assert results.error is None

    def test_search_with_populated_database(self, temp_chroma_path, sample_course, sample_course_chunks):
        """Test searching populated database returns results"""
        # Arrange
        store = VectorStore(temp_chroma_path, "all-MiniLM-L6-v2", 5)
        store.add_course_metadata(sample_course)
        store.add_course_content(sample_course_chunks)

        # Act
        results = store.search(query="computer use")

        # Assert
        assert not results.is_empty()
        assert len(results.documents) > 0
        assert results.error is None

    def test_search_respects_max_results_limit(self, temp_chroma_path, sample_course):
        """Test that search respects max_results parameter"""
        # Arrange
        store = VectorStore(temp_chroma_path, "all-MiniLM-L6-v2", max_results=2)
        store.add_course_metadata(sample_course)

        # Create many chunks
        many_chunks = [
            CourseChunk(
                content=f"Content chunk number {i} about various topics",
                course_title=sample_course.title,
                lesson_number=0,
                chunk_index=i,
                lesson_link=sample_course.lessons[0].lesson_link
            )
            for i in range(10)
        ]
        store.add_course_content(many_chunks)

        # Act
        results = store.search(query="content")

        # Assert
        assert len(results.documents) <= 2

    def test_search_with_custom_limit(self, temp_chroma_path, sample_course):
        """Test search with custom limit parameter"""
        # Arrange
        store = VectorStore(temp_chroma_path, "all-MiniLM-L6-v2", max_results=5)
        store.add_course_metadata(sample_course)

        many_chunks = [
            CourseChunk(
                content=f"Content about topic {i}",
                course_title=sample_course.title,
                lesson_number=0,
                chunk_index=i,
                lesson_link=sample_course.lessons[0].lesson_link
            )
            for i in range(10)
        ]
        store.add_course_content(many_chunks)

        # Act - Override max_results with custom limit
        results = store.search(query="content", limit=3)

        # Assert
        assert len(results.documents) <= 3


class TestVectorStoreCourseFiltering:
    """Test suite for course name filtering"""

    def test_search_with_exact_course_name(self, temp_chroma_path, sample_course, sample_course_chunks):
        """Test searching with exact course name filter"""
        # Arrange
        store = VectorStore(temp_chroma_path, "all-MiniLM-L6-v2", 5)
        store.add_course_metadata(sample_course)
        store.add_course_content(sample_course_chunks)

        # Act
        results = store.search(
            query="introduction",
            course_name=sample_course.title
        )

        # Assert
        assert not results.is_empty()
        for metadata in results.metadata:
            assert metadata['course_title'] == sample_course.title

    def test_search_with_partial_course_name(self, temp_chroma_path, sample_course, sample_course_chunks):
        """Test course name resolution with partial match"""
        # Arrange
        store = VectorStore(temp_chroma_path, "all-MiniLM-L6-v2", 5)
        store.add_course_metadata(sample_course)
        store.add_course_content(sample_course_chunks)

        # Act - Use partial name
        results = store.search(
            query="introduction",
            course_name="Computer Use"  # Partial match
        )

        # Assert - Should still find results via semantic matching
        assert not results.is_empty() or results.error is not None

    def test_search_with_nonexistent_course(self, temp_chroma_path, sample_course, sample_course_chunks):
        """Test searching for nonexistent course returns error"""
        # Arrange
        store = VectorStore(temp_chroma_path, "all-MiniLM-L6-v2", 5)
        store.add_course_metadata(sample_course)
        store.add_course_content(sample_course_chunks)

        # Act
        results = store.search(
            query="test",
            course_name="Completely Nonexistent Course Title XYZ123"
        )

        # Assert
        assert results.error is not None
        assert "No course found" in results.error


class TestVectorStoreLessonFiltering:
    """Test suite for lesson number filtering"""

    def test_search_with_lesson_filter(self, temp_chroma_path, sample_course, sample_course_chunks):
        """Test searching with lesson number filter"""
        # Arrange
        store = VectorStore(temp_chroma_path, "all-MiniLM-L6-v2", 5)
        store.add_course_metadata(sample_course)
        store.add_course_content(sample_course_chunks)

        # Act
        results = store.search(
            query="content",
            lesson_number=0
        )

        # Assert
        assert not results.is_empty()
        for metadata in results.metadata:
            assert metadata['lesson_number'] == 0

    def test_search_with_course_and_lesson_filter(self, temp_chroma_path, sample_course, sample_course_chunks):
        """Test searching with both course and lesson filters"""
        # Arrange
        store = VectorStore(temp_chroma_path, "all-MiniLM-L6-v2", 5)
        store.add_course_metadata(sample_course)
        store.add_course_content(sample_course_chunks)

        # Act
        results = store.search(
            query="content",
            course_name=sample_course.title,
            lesson_number=1
        )

        # Assert
        assert not results.is_empty()
        for metadata in results.metadata:
            assert metadata['course_title'] == sample_course.title
            assert metadata['lesson_number'] == 1


class TestVectorStoreDataManagement:
    """Test suite for adding and managing data"""

    def test_add_course_metadata(self, temp_chroma_path, sample_course):
        """Test adding course metadata"""
        # Arrange
        store = VectorStore(temp_chroma_path, "all-MiniLM-L6-v2", 5)

        # Act
        store.add_course_metadata(sample_course)

        # Assert
        course_titles = store.get_existing_course_titles()
        assert sample_course.title in course_titles

    def test_add_course_content(self, temp_chroma_path, sample_course_chunks):
        """Test adding course content chunks"""
        # Arrange
        store = VectorStore(temp_chroma_path, "all-MiniLM-L6-v2", 5)

        # Act
        store.add_course_content(sample_course_chunks)

        # Assert - Should be able to search for content
        results = store.search(query="computer")
        assert not results.is_empty()

    def test_add_empty_chunks_list(self, temp_chroma_path):
        """Test adding empty chunks list doesn't crash"""
        # Arrange
        store = VectorStore(temp_chroma_path, "all-MiniLM-L6-v2", 5)

        # Act & Assert - Should not raise exception
        store.add_course_content([])

    def test_get_course_count(self, temp_chroma_path, sample_course):
        """Test getting course count"""
        # Arrange
        store = VectorStore(temp_chroma_path, "all-MiniLM-L6-v2", 5)

        # Act - Initially empty
        initial_count = store.get_course_count()
        assert initial_count == 0

        # Add course
        store.add_course_metadata(sample_course)
        final_count = store.get_course_count()

        # Assert
        assert final_count == 1

    def test_get_existing_course_titles(self, temp_chroma_path, sample_course):
        """Test retrieving existing course titles"""
        # Arrange
        store = VectorStore(temp_chroma_path, "all-MiniLM-L6-v2", 5)
        store.add_course_metadata(sample_course)

        # Act
        titles = store.get_existing_course_titles()

        # Assert
        assert len(titles) == 1
        assert sample_course.title in titles

    def test_clear_all_data(self, temp_chroma_path, sample_course, sample_course_chunks):
        """Test clearing all data from vector store"""
        # Arrange
        store = VectorStore(temp_chroma_path, "all-MiniLM-L6-v2", 5)
        store.add_course_metadata(sample_course)
        store.add_course_content(sample_course_chunks)

        # Act
        store.clear_all_data()

        # Assert
        assert store.get_course_count() == 0
        results = store.search(query="computer")
        assert results.is_empty()


class TestSearchResults:
    """Test suite for SearchResults dataclass"""

    def test_from_chroma_with_results(self):
        """Test creating SearchResults from ChromaDB results"""
        # Arrange
        chroma_results = {
            'documents': [['doc1', 'doc2']],
            'metadatas': [[{'key': 'value1'}, {'key': 'value2'}]],
            'distances': [[0.1, 0.2]]
        }

        # Act
        results = SearchResults.from_chroma(chroma_results)

        # Assert
        assert len(results.documents) == 2
        assert len(results.metadata) == 2
        assert len(results.distances) == 2
        assert results.error is None

    def test_from_chroma_with_empty_results(self):
        """Test creating SearchResults from empty ChromaDB results"""
        # Arrange
        chroma_results = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }

        # Act
        results = SearchResults.from_chroma(chroma_results)

        # Assert
        assert results.is_empty()
        assert results.error is None

    def test_empty_with_error(self):
        """Test creating empty SearchResults with error message"""
        # Act
        results = SearchResults.empty("Test error message")

        # Assert
        assert results.is_empty()
        assert results.error == "Test error message"

    def test_is_empty_method(self):
        """Test is_empty() method"""
        # Arrange
        empty_results = SearchResults([], [], [])
        populated_results = SearchResults(['doc'], [{'key': 'val'}], [0.1])

        # Assert
        assert empty_results.is_empty()
        assert not populated_results.is_empty()


class TestVectorStoreErrorHandling:
    """Test suite for error handling"""

    def test_search_with_invalid_filter_handled(self, temp_chroma_path, sample_course, sample_course_chunks):
        """Test that search handles filter errors gracefully"""
        # Arrange
        store = VectorStore(temp_chroma_path, "all-MiniLM-L6-v2", 5)
        store.add_course_metadata(sample_course)
        store.add_course_content(sample_course_chunks)

        # Act - This should be handled gracefully
        results = store.search(query="test", course_name="Nonexistent")

        # Assert - Should return error or empty results, not crash
        assert results.error is not None or results.is_empty()

    def test_get_course_count_on_error_returns_zero(self, temp_chroma_path):
        """Test that get_course_count returns 0 on error"""
        # Arrange
        store = VectorStore(temp_chroma_path, "all-MiniLM-L6-v2", 5)

        # Simulate error condition by deleting collection
        store.client.delete_collection("course_catalog")

        # Act
        count = store.get_course_count()

        # Assert
        assert count == 0

    def test_get_existing_titles_on_error_returns_empty_list(self, temp_chroma_path):
        """Test that get_existing_course_titles returns empty list on error"""
        # Arrange
        store = VectorStore(temp_chroma_path, "all-MiniLM-L6-v2", 5)

        # Simulate error condition
        store.client.delete_collection("course_catalog")

        # Act
        titles = store.get_existing_course_titles()

        # Assert
        assert titles == []
