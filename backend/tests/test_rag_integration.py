"""
Integration tests for the complete RAG system

These tests verify end-to-end functionality:
1. Complete query flow from RAG system through all components
2. Session management with conversation history
3. Empty vs populated vector store scenarios
4. Real-world query scenarios
5. Error propagation through the system
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil

# Add backend directory to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from rag_system import RAGSystem
from config import Config


class TestRAGSystemInitialization:
    """Test suite for RAG system initialization"""

    def test_rag_system_initializes_all_components(self, mock_config):
        """Test that RAG system initializes all required components"""
        # Act
        with patch('rag_system.VectorStore'), \
             patch('rag_system.AIGenerator'), \
             patch('rag_system.DocumentProcessor'):
            rag = RAGSystem(mock_config)

        # Assert
        assert rag.vector_store is not None
        assert rag.ai_generator is not None
        assert rag.document_processor is not None
        assert rag.session_manager is not None
        assert rag.tool_manager is not None
        assert rag.search_tool is not None

    def test_search_tool_registered_in_tool_manager(self, mock_config):
        """Test that search tool is registered during initialization"""
        # Act
        with patch('rag_system.VectorStore'), \
             patch('rag_system.AIGenerator'), \
             patch('rag_system.DocumentProcessor'):
            rag = RAGSystem(mock_config)

        # Assert
        tool_definitions = rag.tool_manager.get_tool_definitions()
        assert len(tool_definitions) == 1
        assert tool_definitions[0]['name'] == 'search_course_content'


class TestRAGSystemQueryWithEmptyVectorStore:
    """Test RAG system behavior when vector store is empty"""

    def test_query_with_empty_vector_store_still_responds(self, mock_config):
        """Test that system handles queries even when vector store is empty"""
        # Arrange - Mock all components
        with patch('rag_system.VectorStore') as MockVectorStore, \
             patch('rag_system.AIGenerator') as MockAIGenerator, \
             patch('rag_system.DocumentProcessor'):

            # Configure mocks
            mock_vector_store = MockVectorStore.return_value
            mock_vector_store.search.return_value = Mock(
                documents=[],
                metadata=[],
                distances=[],
                error=None,
                is_empty=lambda: True
            )

            mock_ai = MockAIGenerator.return_value
            mock_ai.generate_response.return_value = "I couldn't find any relevant course content."

            rag = RAGSystem(mock_config)

            # Act
            response, sources = rag.query("What is computer use?")

            # Assert
            assert response is not None
            assert isinstance(response, str)
            assert isinstance(sources, list)

    def test_query_with_no_api_key_fails(self):
        """Test that query fails gracefully when API key is missing"""
        # Arrange
        config = Mock()
        config.ANTHROPIC_API_KEY = ""  # Empty API key
        config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
        config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        config.CHUNK_SIZE = 800
        config.CHUNK_OVERLAP = 100
        config.MAX_RESULTS = 5
        config.MAX_HISTORY = 2
        config.CHROMA_PATH = "./test_chroma"

        # Act & Assert - Should raise error when making API call
        with patch('rag_system.VectorStore'), \
             patch('rag_system.DocumentProcessor'):
            rag = RAGSystem(config)

            # Calling query should eventually fail when reaching Anthropic API
            with pytest.raises(Exception):
                rag.query("test query")


class TestRAGSystemQueryWithPopulatedStore:
    """Test RAG system with populated vector store"""

    def test_query_triggers_search_and_returns_response(self, mock_config, sample_search_results):
        """Test complete query flow with successful search"""
        # Arrange
        with patch('rag_system.VectorStore') as MockVectorStore, \
             patch('rag_system.AIGenerator') as MockAIGenerator, \
             patch('rag_system.DocumentProcessor'):

            # Mock vector store with results
            mock_vector_store = MockVectorStore.return_value
            mock_vector_store.search.return_value = sample_search_results

            # Mock AI that calls tool then responds
            mock_ai = MockAIGenerator.return_value
            mock_ai.generate_response.return_value = "Computer use allows Claude to interact with computers through screenshots and actions."

            rag = RAGSystem(mock_config)

            # Act
            response, sources = rag.query("What is computer use?")

            # Assert
            assert response is not None
            assert "computer" in response.lower()
            assert mock_ai.generate_response.called

    def test_query_returns_sources_from_search(self, mock_config, sample_search_results):
        """Test that query returns sources from tool search"""
        # Arrange
        with patch('rag_system.VectorStore') as MockVectorStore, \
             patch('rag_system.AIGenerator') as MockAIGenerator, \
             patch('rag_system.DocumentProcessor'):

            mock_vector_store = MockVectorStore.return_value

            # Create a realistic flow: AI calls tool, tool executes search
            def ai_generate_side_effect(query, conversation_history, tools, tool_manager):
                # Simulate AI requesting tool use
                tool_manager.execute_tool('search_course_content', query="computer use")
                return "Response based on search results"

            mock_ai = MockAIGenerator.return_value
            mock_ai.generate_response.side_effect = ai_generate_side_effect

            # Configure vector store
            mock_vector_store.search.return_value = sample_search_results

            rag = RAGSystem(mock_config)

            # Act
            response, sources = rag.query("What is computer use?")

            # Assert
            assert len(sources) > 0
            assert sources[0]['text'] == "Building Towards Computer Use with Anthropic - Lesson 0"

    def test_query_with_session_id_uses_history(self, mock_config):
        """Test that providing session_id includes conversation history"""
        # Arrange
        with patch('rag_system.VectorStore') as MockVectorStore, \
             patch('rag_system.AIGenerator') as MockAIGenerator, \
             patch('rag_system.DocumentProcessor'):

            mock_ai = MockAIGenerator.return_value
            mock_ai.generate_response.return_value = "Follow-up response"

            mock_vector_store = MockVectorStore.return_value
            mock_vector_store.search.return_value = Mock(
                documents=[],
                metadata=[],
                distances=[],
                error=None,
                is_empty=lambda: True
            )

            rag = RAGSystem(mock_config)

            # Act - Create session and make multiple queries
            session_id = rag.session_manager.create_session()
            rag.query("First question", session_id)
            rag.query("Follow-up question", session_id)

            # Assert - Second call should include history
            calls = mock_ai.generate_response.call_args_list
            assert len(calls) == 2
            second_call_kwargs = calls[1][1]
            assert second_call_kwargs['conversation_history'] is not None


class TestRAGSystemDocumentProcessing:
    """Test RAG system document processing functionality"""

    def test_add_course_document_processes_and_stores(self, mock_config, sample_course, sample_course_chunks):
        """Test adding a course document"""
        # Arrange
        with patch('rag_system.VectorStore') as MockVectorStore, \
             patch('rag_system.AIGenerator'), \
             patch('rag_system.DocumentProcessor') as MockDocProcessor:

            mock_vector_store = MockVectorStore.return_value
            mock_doc_processor = MockDocProcessor.return_value
            mock_doc_processor.process_course_document.return_value = (sample_course, sample_course_chunks)

            rag = RAGSystem(mock_config)

            # Act
            course, chunk_count = rag.add_course_document("/path/to/course.txt")

            # Assert
            assert course == sample_course
            assert chunk_count == len(sample_course_chunks)
            mock_vector_store.add_course_metadata.assert_called_once_with(sample_course)
            mock_vector_store.add_course_content.assert_called_once_with(sample_course_chunks)

    def test_add_course_folder_processes_multiple_files(self, mock_config, sample_course, sample_course_chunks):
        """Test adding courses from a folder"""
        # Arrange
        with patch('rag_system.VectorStore') as MockVectorStore, \
             patch('rag_system.AIGenerator'), \
             patch('rag_system.DocumentProcessor') as MockDocProcessor, \
             patch('os.path.exists', return_value=True), \
             patch('os.listdir', return_value=['course1.txt', 'course2.txt']), \
             patch('os.path.isfile', return_value=True):

            mock_vector_store = MockVectorStore.return_value
            mock_vector_store.get_existing_course_titles.return_value = []

            mock_doc_processor = MockDocProcessor.return_value
            mock_doc_processor.process_course_document.return_value = (sample_course, sample_course_chunks)

            rag = RAGSystem(mock_config)

            # Act
            total_courses, total_chunks = rag.add_course_folder("/path/to/docs")

            # Assert
            assert total_courses == 2
            assert total_chunks == len(sample_course_chunks) * 2

    def test_add_course_folder_skips_existing_courses(self, mock_config, sample_course, sample_course_chunks):
        """Test that existing courses are not re-added"""
        # Arrange
        with patch('rag_system.VectorStore') as MockVectorStore, \
             patch('rag_system.AIGenerator'), \
             patch('rag_system.DocumentProcessor') as MockDocProcessor, \
             patch('os.path.exists', return_value=True), \
             patch('os.listdir', return_value=['course1.txt']), \
             patch('os.path.isfile', return_value=True):

            mock_vector_store = MockVectorStore.return_value
            # Course already exists
            mock_vector_store.get_existing_course_titles.return_value = [sample_course.title]

            mock_doc_processor = MockDocProcessor.return_value
            mock_doc_processor.process_course_document.return_value = (sample_course, sample_course_chunks)

            rag = RAGSystem(mock_config)

            # Act
            total_courses, total_chunks = rag.add_course_folder("/path/to/docs")

            # Assert
            assert total_courses == 0  # Should skip existing course
            assert total_chunks == 0


class TestRAGSystemAnalytics:
    """Test RAG system analytics functionality"""

    def test_get_course_analytics_returns_stats(self, mock_config):
        """Test getting course analytics"""
        # Arrange
        with patch('rag_system.VectorStore') as MockVectorStore, \
             patch('rag_system.AIGenerator'), \
             patch('rag_system.DocumentProcessor'):

            mock_vector_store = MockVectorStore.return_value
            mock_vector_store.get_course_count.return_value = 3
            mock_vector_store.get_existing_course_titles.return_value = [
                "Course 1", "Course 2", "Course 3"
            ]

            rag = RAGSystem(mock_config)

            # Act
            analytics = rag.get_course_analytics()

            # Assert
            assert analytics['total_courses'] == 3
            assert len(analytics['course_titles']) == 3


class TestRAGSystemSessionManagement:
    """Test session management in RAG system"""

    def test_query_without_session_id_creates_session(self, mock_config):
        """Test that query can be made without providing session_id"""
        # Arrange
        with patch('rag_system.VectorStore') as MockVectorStore, \
             patch('rag_system.AIGenerator') as MockAIGenerator, \
             patch('rag_system.DocumentProcessor'):

            mock_ai = MockAIGenerator.return_value
            mock_ai.generate_response.return_value = "Response"

            mock_vector_store = MockVectorStore.return_value
            mock_vector_store.search.return_value = Mock(
                documents=[],
                metadata=[],
                distances=[],
                error=None,
                is_empty=lambda: True
            )

            rag = RAGSystem(mock_config)

            # Act - No session_id provided
            response, sources = rag.query("Test question")

            # Assert
            assert response is not None

    def test_conversation_history_updated_after_query(self, mock_config):
        """Test that conversation history is updated after each query"""
        # Arrange
        with patch('rag_system.VectorStore') as MockVectorStore, \
             patch('rag_system.AIGenerator') as MockAIGenerator, \
             patch('rag_system.DocumentProcessor'):

            mock_ai = MockAIGenerator.return_value
            mock_ai.generate_response.return_value = "Response"

            mock_vector_store = MockVectorStore.return_value
            mock_vector_store.search.return_value = Mock(
                documents=[],
                metadata=[],
                distances=[],
                error=None,
                is_empty=lambda: True
            )

            rag = RAGSystem(mock_config)

            # Act
            session_id = rag.session_manager.create_session()
            rag.query("Question 1", session_id)
            rag.query("Question 2", session_id)

            # Assert
            history = rag.session_manager.get_conversation_history(session_id)
            assert "Question 1" in history
            assert "Question 2" in history


class TestRAGSystemErrorScenarios:
    """Test error handling in RAG system"""

    def test_document_processing_error_handled(self, mock_config):
        """Test that document processing errors are handled gracefully"""
        # Arrange
        with patch('rag_system.VectorStore') as MockVectorStore, \
             patch('rag_system.AIGenerator'), \
             patch('rag_system.DocumentProcessor') as MockDocProcessor:

            mock_doc_processor = MockDocProcessor.return_value
            mock_doc_processor.process_course_document.side_effect = Exception("Parse error")

            rag = RAGSystem(mock_config)

            # Act
            course, chunks = rag.add_course_document("/bad/path.txt")

            # Assert
            assert course is None
            assert chunks == 0

    def test_query_with_vector_store_error(self, mock_config):
        """Test query when vector store returns error"""
        # Arrange
        with patch('rag_system.VectorStore') as MockVectorStore, \
             patch('rag_system.AIGenerator') as MockAIGenerator, \
             patch('rag_system.DocumentProcessor'):

            # Mock vector store that returns error
            mock_vector_store = MockVectorStore.return_value
            error_result = Mock(
                documents=[],
                metadata=[],
                distances=[],
                error="Database connection failed",
                is_empty=lambda: True
            )
            mock_vector_store.search.return_value = error_result

            # Mock AI to handle the error result
            mock_ai = MockAIGenerator.return_value

            def ai_side_effect(query, conversation_history, tools, tool_manager):
                # Simulate tool execution that returns error
                result = tool_manager.execute_tool('search_course_content', query="test")
                # AI should get the error message and respond accordingly
                return "I encountered an error searching the course content."

            mock_ai.generate_response.side_effect = ai_side_effect

            rag = RAGSystem(mock_config)

            # Act
            response, sources = rag.query("Test query")

            # Assert - Should handle gracefully
            assert response is not None
            assert isinstance(response, str)


class TestRAGSystemRealWorldScenarios:
    """Test realistic usage scenarios"""

    def test_typical_user_query_flow(self, mock_config, sample_search_results):
        """Test a typical user query from start to finish"""
        # Arrange
        with patch('rag_system.VectorStore') as MockVectorStore, \
             patch('rag_system.AIGenerator') as MockAIGenerator, \
             patch('rag_system.DocumentProcessor'):

            mock_vector_store = MockVectorStore.return_value

            # Simulate realistic AI tool calling behavior
            def realistic_ai_behavior(query, conversation_history, tools, tool_manager):
                # AI decides to use search tool
                search_results = tool_manager.execute_tool(
                    'search_course_content',
                    query="computer use"
                )
                # AI synthesizes response from search results
                return "Computer use is a feature that allows Claude to interact with computers."

            mock_ai = MockAIGenerator.return_value
            mock_ai.generate_response.side_effect = realistic_ai_behavior

            mock_vector_store.search.return_value = sample_search_results

            rag = RAGSystem(mock_config)

            # Act - User asks question
            response, sources = rag.query("What is computer use in Claude?")

            # Assert
            assert response is not None
            assert len(sources) > 0
            assert sources[0]['link'] is not None

    def test_multi_turn_conversation(self, mock_config):
        """Test multi-turn conversation maintains context"""
        # Arrange
        with patch('rag_system.VectorStore') as MockVectorStore, \
             patch('rag_system.AIGenerator') as MockAIGenerator, \
             patch('rag_system.DocumentProcessor'):

            mock_ai = MockAIGenerator.return_value
            responses = [
                "Tool use allows Claude to call external functions.",
                "Yes, you can use multiple tools in one request.",
                "Tools are defined using JSON schema."
            ]
            mock_ai.generate_response.side_effect = responses

            mock_vector_store = MockVectorStore.return_value
            mock_vector_store.search.return_value = Mock(
                documents=["Tool use content"],
                metadata=[{'course_title': 'Test', 'lesson_number': 1, 'lesson_link': 'http://test.com'}],
                distances=[0.1],
                error=None,
                is_empty=lambda: False
            )

            rag = RAGSystem(mock_config)
            session_id = rag.session_manager.create_session()

            # Act - Multi-turn conversation
            response1, _ = rag.query("What is tool use?", session_id)
            response2, _ = rag.query("Can I use multiple tools?", session_id)
            response3, _ = rag.query("How do I define them?", session_id)

            # Assert
            assert "tool" in response1.lower()
            assert "multiple" in response2.lower()
            assert "schema" in response3.lower()
            # History should be passed to all calls after the first
            assert mock_ai.generate_response.call_count == 3
