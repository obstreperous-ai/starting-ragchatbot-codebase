"""
Tests for AIGenerator and its tool calling functionality

These tests verify:
1. Basic response generation without tools
2. Tool calling flow when Claude requests search
3. Conversation history handling
4. Error scenarios and edge cases
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add backend directory to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from ai_generator import AIGenerator
from search_tools import ToolManager, CourseSearchTool


class TestAIGeneratorBasicResponses:
    """Test suite for basic AIGenerator response generation"""

    def test_generate_simple_response_without_tools(self, mock_anthropic_client):
        """Test generating a response without tool use"""
        # Arrange
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")

        # Act
        response = generator.generate_response(query="What is AI?")

        # Assert
        assert response == "This is a test response"
        mock_anthropic_client.messages.create.assert_called_once()

    def test_generate_response_with_conversation_history(self, mock_anthropic_client):
        """Test response generation includes conversation history in system prompt"""
        # Arrange
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
        history = "User: Previous question\nAssistant: Previous answer"

        # Act
        response = generator.generate_response(query="Follow-up question", conversation_history=history)

        # Assert
        call_kwargs = mock_anthropic_client.messages.create.call_args[1]
        assert "Previous question" in call_kwargs['system']
        assert "Previous answer" in call_kwargs['system']

    def test_generate_response_passes_correct_parameters(self, mock_anthropic_client):
        """Test that correct parameters are passed to Anthropic API"""
        # Arrange
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")

        # Act
        generator.generate_response(query="Test query")

        # Assert
        call_kwargs = mock_anthropic_client.messages.create.call_args[1]
        assert call_kwargs['model'] == "claude-sonnet-4-20250514"
        assert call_kwargs['temperature'] == 0
        assert call_kwargs['max_tokens'] == 800
        assert len(call_kwargs['messages']) == 1
        assert call_kwargs['messages'][0]['role'] == 'user'
        assert "Test query" in call_kwargs['messages'][0]['content']


class TestAIGeneratorToolCalling:
    """Test suite for AIGenerator tool calling functionality"""

    def test_generate_response_with_tools_provided(self, mock_anthropic_client, mock_vector_store):
        """Test that tools are passed to API when provided"""
        # Arrange
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")

        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        tool_manager.register_tool(search_tool)
        tools = tool_manager.get_tool_definitions()

        # Act
        generator.generate_response(query="Test", tools=tools)

        # Assert
        call_kwargs = mock_anthropic_client.messages.create.call_args[1]
        assert 'tools' in call_kwargs
        assert len(call_kwargs['tools']) == 1
        assert call_kwargs['tool_choice'] == {"type": "auto"}

    def test_tool_execution_flow(self, mock_anthropic_client_with_tool_use, populated_mock_vector_store, sample_search_results):
        """Test complete tool execution flow: request → execute → final response"""
        # Arrange
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client_with_tool_use):
            generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")

        populated_mock_vector_store.search.return_value = sample_search_results
        tool_manager = ToolManager()
        search_tool = CourseSearchTool(populated_mock_vector_store)
        tool_manager.register_tool(search_tool)
        tools = tool_manager.get_tool_definitions()

        # Act
        response = generator.generate_response(
            query="What is computer use?",
            tools=tools,
            tool_manager=tool_manager
        )

        # Assert - Should get final response after tool execution
        assert "computer use allows Claude to interact" in response
        # Verify API was called twice (tool request + final response)
        assert mock_anthropic_client_with_tool_use.messages.create.call_count == 2
        # Verify search was executed
        populated_mock_vector_store.search.assert_called_once()

    def test_tool_result_passed_back_to_api(self, mock_anthropic_client_with_tool_use, populated_mock_vector_store, sample_search_results):
        """Test that tool execution results are passed back to Claude"""
        # Arrange
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client_with_tool_use):
            generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")

        populated_mock_vector_store.search.return_value = sample_search_results
        tool_manager = ToolManager()
        search_tool = CourseSearchTool(populated_mock_vector_store)
        tool_manager.register_tool(search_tool)
        tools = tool_manager.get_tool_definitions()

        # Act
        generator.generate_response(
            query="What is computer use?",
            tools=tools,
            tool_manager=tool_manager
        )

        # Assert - Check second API call includes tool results
        second_call_kwargs = mock_anthropic_client_with_tool_use.messages.create.call_args_list[1][1]
        messages = second_call_kwargs['messages']

        # Should have: original user message, assistant tool use, user tool result
        assert len(messages) >= 3
        # Last message should be tool results
        assert messages[-1]['role'] == 'user'
        assert 'content' in messages[-1]
        # Tool results should be a list with tool_result type
        tool_results = messages[-1]['content']
        assert isinstance(tool_results, list)
        assert tool_results[0]['type'] == 'tool_result'
        assert tool_results[0]['tool_use_id'] == 'tool_123'

    def test_no_tool_execution_without_tool_manager(self, populated_mock_vector_store):
        """Test that tool use is skipped if no tool_manager provided"""
        # Arrange - Create a mock client that returns tool_use
        mock_client = Mock()
        tool_use_block = Mock()
        tool_use_block.type = "tool_use"
        tool_use_block.id = "tool_123"
        tool_use_block.name = "search_course_content"
        tool_use_block.input = {"query": "test"}

        tool_response = Mock()
        tool_response.stop_reason = "tool_use"
        tool_response.content = [tool_use_block]
        mock_client.messages.create.return_value = tool_response

        with patch('ai_generator.anthropic.Anthropic', return_value=mock_client):
            generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")

        tool_manager = ToolManager()
        search_tool = CourseSearchTool(populated_mock_vector_store)
        tool_manager.register_tool(search_tool)
        tools = tool_manager.get_tool_definitions()

        # Act - Pass tools but no tool_manager
        response = generator.generate_response(query="Test", tools=tools, tool_manager=None)

        # Assert - Should return None or handle gracefully (no second API call)
        assert mock_client.messages.create.call_count == 1

    def test_multiple_tool_calls_in_single_response(self, populated_mock_vector_store, sample_search_results):
        """Test handling multiple tool calls in a single response"""
        # Arrange - Mock client with multiple tool uses
        mock_client = Mock()

        # Create two tool use blocks
        tool_use_1 = Mock()
        tool_use_1.type = "tool_use"
        tool_use_1.id = "tool_1"
        tool_use_1.name = "search_course_content"
        tool_use_1.input = {"query": "first query"}

        tool_use_2 = Mock()
        tool_use_2.type = "tool_use"
        tool_use_2.id = "tool_2"
        tool_use_2.name = "search_course_content"
        tool_use_2.input = {"query": "second query"}

        tool_response = Mock()
        tool_response.stop_reason = "tool_use"
        tool_response.content = [tool_use_1, tool_use_2]

        final_response = Mock()
        final_response.stop_reason = "end_turn"
        final_response.content = [Mock(text="Combined answer", type="text")]

        mock_client.messages.create.side_effect = [tool_response, final_response]

        with patch('ai_generator.anthropic.Anthropic', return_value=mock_client):
            generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")

        populated_mock_vector_store.search.return_value = sample_search_results
        tool_manager = ToolManager()
        search_tool = CourseSearchTool(populated_mock_vector_store)
        tool_manager.register_tool(search_tool)
        tools = tool_manager.get_tool_definitions()

        # Act
        response = generator.generate_response(
            query="Complex query",
            tools=tools,
            tool_manager=tool_manager
        )

        # Assert
        assert response == "Combined answer"
        # Should execute search twice
        assert populated_mock_vector_store.search.call_count == 2


class TestAIGeneratorErrorHandling:
    """Test suite for AIGenerator error handling"""

    def test_api_error_propagates(self):
        """Test that API errors are propagated to caller"""
        # Arrange
        mock_client = Mock()
        mock_client.messages.create.side_effect = Exception("API Error")

        with patch('ai_generator.anthropic.Anthropic', return_value=mock_client):
            generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")

        # Act & Assert
        with pytest.raises(Exception) as exc_info:
            generator.generate_response(query="Test")
        assert "API Error" in str(exc_info.value)

    def test_invalid_tool_name_handled(self, mock_anthropic_client_with_tool_use, mock_vector_store):
        """Test handling of tool call with invalid tool name"""
        # Arrange - Modify tool use to request non-existent tool
        tool_use_block = mock_anthropic_client_with_tool_use.messages.create.side_effect[0].content[0]
        tool_use_block.name = "nonexistent_tool"

        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client_with_tool_use):
            generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")

        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        tool_manager.register_tool(search_tool)
        tools = tool_manager.get_tool_definitions()

        # Act
        response = generator.generate_response(
            query="Test",
            tools=tools,
            tool_manager=tool_manager
        )

        # Assert - Should complete without crashing
        # Tool manager returns "Tool 'X' not found" message
        assert response is not None

    def test_empty_api_key_initialization(self):
        """Test initialization with empty API key"""
        # This should work - error will occur when making actual API calls
        generator = AIGenerator(api_key="", model="claude-sonnet-4-20250514")
        assert generator.model == "claude-sonnet-4-20250514"


class TestAIGeneratorSystemPrompt:
    """Test suite for system prompt configuration"""

    def test_system_prompt_includes_instructions(self, mock_anthropic_client):
        """Test that system prompt includes search tool usage instructions"""
        # Arrange
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")

        # Act
        generator.generate_response(query="Test")

        # Assert
        call_kwargs = mock_anthropic_client.messages.create.call_args[1]
        system_prompt = call_kwargs['system']
        assert "course materials" in system_prompt.lower()
        assert "search" in system_prompt.lower()

    def test_system_prompt_consistency(self, mock_anthropic_client):
        """Test that system prompt is consistent across calls"""
        # Arrange
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")

        # Act - Make two calls
        generator.generate_response(query="First query")
        first_system = mock_anthropic_client.messages.create.call_args[1]['system']

        generator.generate_response(query="Second query")
        second_system = mock_anthropic_client.messages.create.call_args[1]['system']

        # Assert - Without history, system prompts should be identical
        assert first_system == second_system
