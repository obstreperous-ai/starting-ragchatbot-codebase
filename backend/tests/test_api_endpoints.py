"""
API endpoint tests for the FastAPI application

These tests verify the REST API endpoints work correctly.
We create a test app that doesn't mount static files to avoid filesystem dependencies.
"""
import pytest
from unittest.mock import Mock
from fastapi.testclient import TestClient
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional


# Define request/response models (mirrored from app.py)
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class SourceItem(BaseModel):
    text: str
    link: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceItem]
    session_id: str


class CourseStats(BaseModel):
    total_courses: int
    course_titles: List[str]


@pytest.fixture
def test_app(mock_rag_system):
    """Create a test FastAPI app without static file mounting"""
    app = FastAPI(title="Course Materials RAG System - Test")

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Inject mock RAG system
    app.state.rag_system = mock_rag_system

    # Define endpoints (same as app.py but without static mounting)
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = app.state.rag_system.session_manager.create_session()

            answer, sources = app.state.rag_system.query(request.query, session_id)

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
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = app.state.rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


@pytest.fixture
def client(test_app):
    """Create a test client for the FastAPI app"""
    return TestClient(test_app)


@pytest.mark.api
class TestQueryEndpoint:
    """Tests for /api/query endpoint"""

    def test_query_with_session_id(self, client, sample_query_request):
        """Test successful query with existing session ID"""
        response = client.post("/api/query", json=sample_query_request)

        assert response.status_code == 200
        data = response.json()

        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["session_id"] == "test-session-123"
        assert isinstance(data["sources"], list)
        assert len(data["sources"]) > 0

    def test_query_without_session_id(self, client, sample_query_request_no_session):
        """Test query creates new session when none provided"""
        response = client.post("/api/query", json=sample_query_request_no_session)

        assert response.status_code == 200
        data = response.json()

        assert "session_id" in data
        assert data["session_id"] == "test-session-123"

    def test_query_validates_request_body(self, client):
        """Test query endpoint validates request body"""
        # Missing required field 'query'
        response = client.post("/api/query", json={})

        assert response.status_code == 422  # Unprocessable Entity

    def test_query_with_empty_query(self, client):
        """Test query with empty string"""
        response = client.post("/api/query", json={"query": ""})

        assert response.status_code == 200  # FastAPI allows empty strings

    def test_query_returns_proper_structure(self, client, sample_query_request):
        """Test response follows QueryResponse schema"""
        response = client.post("/api/query", json=sample_query_request)

        assert response.status_code == 200
        data = response.json()

        # Validate structure
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)

        # Validate source structure
        for source in data["sources"]:
            assert "text" in source
            assert isinstance(source["text"], str)
            # link can be null/None
            if source.get("link"):
                assert isinstance(source["link"], str)

    def test_query_handles_rag_system_error(self, client, test_app, sample_query_request):
        """Test error handling when RAG system fails"""
        # Mock RAG system to raise exception
        test_app.state.rag_system.query.side_effect = Exception("Database connection failed")

        response = client.post("/api/query", json=sample_query_request)

        assert response.status_code == 500
        assert "Database connection failed" in response.json()["detail"]

    def test_query_with_special_characters(self, client):
        """Test query handles special characters"""
        special_query = {
            "query": "What is <script>alert('test')</script> in computer use?",
            "session_id": "test-session"
        }

        response = client.post("/api/query", json=special_query)

        assert response.status_code == 200

    def test_query_with_long_text(self, client):
        """Test query handles long input text"""
        long_query = {
            "query": "What is computer use? " * 100,
            "session_id": "test-session"
        }

        response = client.post("/api/query", json=long_query)

        assert response.status_code == 200

    def test_query_sources_without_links(self, client, test_app, sample_query_request):
        """Test handling sources that don't have links"""
        # Mock response with sources without links
        test_app.state.rag_system.query.return_value = (
            "Test answer",
            [
                {"text": "Source without link"},
                {"text": "Another source", "link": None}
            ]
        )

        response = client.post("/api/query", json=sample_query_request)

        assert response.status_code == 200
        data = response.json()
        assert len(data["sources"]) == 2
        # Both sources should have None/null links
        assert data["sources"][0]["link"] is None
        assert data["sources"][1]["link"] is None


@pytest.mark.api
class TestCoursesEndpoint:
    """Tests for /api/courses endpoint"""

    def test_get_courses_success(self, client):
        """Test successful course statistics retrieval"""
        response = client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()

        assert "total_courses" in data
        assert "course_titles" in data
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)

    def test_get_courses_returns_correct_data(self, client, test_app):
        """Test courses endpoint returns expected data structure"""
        # Mock specific analytics data
        test_app.state.rag_system.get_course_analytics.return_value = {
            "total_courses": 3,
            "course_titles": ["Course A", "Course B", "Course C"]
        }

        response = client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()

        assert data["total_courses"] == 3
        assert len(data["course_titles"]) == 3
        assert "Course A" in data["course_titles"]

    def test_get_courses_handles_empty_data(self, client, test_app):
        """Test courses endpoint with no courses"""
        test_app.state.rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": []
        }

        response = client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()

        assert data["total_courses"] == 0
        assert data["course_titles"] == []

    def test_get_courses_handles_error(self, client, test_app):
        """Test error handling when getting course analytics fails"""
        test_app.state.rag_system.get_course_analytics.side_effect = Exception("Vector store error")

        response = client.get("/api/courses")

        assert response.status_code == 500
        assert "Vector store error" in response.json()["detail"]

    def test_get_courses_no_parameters(self, client):
        """Test courses endpoint doesn't accept parameters"""
        response = client.get("/api/courses?invalid_param=test")

        # Should still work, just ignore the parameter
        assert response.status_code == 200


@pytest.mark.api
class TestCORSAndMiddleware:
    """Tests for CORS and middleware configuration"""

    def test_middleware_configured(self, test_app):
        """Test middleware is configured in the app"""
        # Verify that middleware stack is not empty
        # The preflight test verifies CORS actually works
        assert len(test_app.user_middleware) > 0, "Middleware should be configured"

    def test_preflight_request(self, client):
        """Test OPTIONS preflight request works"""
        response = client.options(
            "/api/query",
            headers={
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type",
                "Origin": "http://localhost:3000"
            }
        )

        assert response.status_code == 200


@pytest.mark.api
class TestEndpointIntegration:
    """Integration tests across multiple endpoints"""

    def test_query_then_courses(self, client, sample_query_request):
        """Test making a query followed by getting courses"""
        # First query
        query_response = client.post("/api/query", json=sample_query_request)
        assert query_response.status_code == 200

        # Then get courses
        courses_response = client.get("/api/courses")
        assert courses_response.status_code == 200

    def test_multiple_queries_same_session(self, client):
        """Test multiple queries with the same session ID"""
        session_id = "consistent-session-id"

        # First query
        response1 = client.post("/api/query", json={
            "query": "What is computer use?",
            "session_id": session_id
        })
        assert response1.status_code == 200
        assert response1.json()["session_id"] == session_id

        # Second query with same session
        response2 = client.post("/api/query", json={
            "query": "Tell me more about tools",
            "session_id": session_id
        })
        assert response2.status_code == 200
        assert response2.json()["session_id"] == session_id

    def test_concurrent_sessions(self, client):
        """Test different session IDs are handled independently"""
        response1 = client.post("/api/query", json={
            "query": "Query 1",
            "session_id": "session-1"
        })

        response2 = client.post("/api/query", json={
            "query": "Query 2",
            "session_id": "session-2"
        })

        assert response1.status_code == 200
        assert response2.status_code == 200
        assert response1.json()["session_id"] != response2.json()["session_id"]


@pytest.mark.api
class TestErrorHandling:
    """Tests for error handling across endpoints"""

    def test_invalid_json_body(self, client):
        """Test handling of malformed JSON"""
        response = client.post(
            "/api/query",
            data="invalid json{",
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422

    def test_wrong_content_type(self, client):
        """Test with incorrect content type"""
        response = client.post(
            "/api/query",
            data="query=test",
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )

        assert response.status_code == 422

    def test_invalid_endpoint(self, client):
        """Test accessing non-existent endpoint"""
        response = client.get("/api/nonexistent")

        assert response.status_code == 404

    def test_wrong_method(self, client):
        """Test using wrong HTTP method"""
        # GET instead of POST for /api/query
        response = client.get("/api/query")

        assert response.status_code == 405  # Method Not Allowed

        # POST instead of GET for /api/courses
        response = client.post("/api/courses", json={})

        assert response.status_code == 405
