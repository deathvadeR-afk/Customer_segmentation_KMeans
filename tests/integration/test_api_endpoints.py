"""Integration tests for API endpoints."""
import pytest


class TestPredictEndpoint:
    """Tests for /predict endpoint."""

    def test_predict_success(self, test_client):
        """Test successful prediction."""
        response = test_client.post(
            "/predict",
            json={"annual_income": 60.0, "spending_score": 50.0}
        )

        assert response.status_code == 200
        data = response.json()
        assert "cluster_id" in data
        assert "cluster_name" in data
        assert "request_id" in data
        assert 0 <= data["cluster_id"] <= 4

    def test_predict_validation_error(self, test_client):
        """Test prediction with invalid input."""
        response = test_client.post(
            "/predict",
            json={"annual_income": 200.0, "spending_score": 50.0}
        )

        assert response.status_code == 422

    def test_predict_missing_fields(self, test_client):
        """Test prediction with missing fields."""
        response = test_client.post(
            "/predict",
            json={"annual_income": 60.0}
        )

        assert response.status_code == 422


class TestBatchPredictEndpoint:
    """Tests for /predict/batch endpoint."""

    def test_batch_predict_success(self, test_client):
        """Test successful batch prediction."""
        response = test_client.post(
            "/predict/batch",
            json={
                "customers": [
                    {"annual_income": 60.0, "spending_score": 50.0},
                    {"annual_income": 80.0, "spending_score": 70.0},
                    {"annual_income": 40.0, "spending_score": 30.0},
                ]
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert "total_count" in data
        assert len(data["predictions"]) == 3
        assert data["total_count"] == 3

    def test_batch_predict_size_limit(self, test_client):
        """Test batch prediction exceeds size limit."""
        # Config has batch_max_size = 10 for test
        customers = [
            {"annual_income": 60.0, "spending_score": 50.0}
            for _ in range(15)
        ]

        response = test_client.post(
            "/predict/batch",
            json={"customers": customers}
        )

        # Should fail validation (pydantic max_length=100 in schema,
        # but predictor has batch_max_size=10)
        assert response.status_code in [422, 500]


class TestRetrainEndpoint:
    """Tests for /retrain endpoint."""

    def test_retrain_no_api_key(self, test_client):
        """Test retrain without API key."""
        response = test_client.post(
            "/retrain",
            json={"n_clusters": 5}
        )

        assert response.status_code == 422  # Missing header

    def test_retrain_invalid_api_key(self, test_client):
        """Test retrain with invalid API key."""
        response = test_client.post(
            "/retrain",
            json={"n_clusters": 5},
            headers={"X-API-Key": "wrong_key"}
        )

        assert response.status_code == 401

    def test_retrain_success(self, test_client):
        """Test retrain with valid API key."""
        response = test_client.post(
            "/retrain",
            json={"n_clusters": 5},
            headers={"X-API-Key": "test_api_key"}
        )

        # Should accept the request
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "started"


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_check(self, test_client):
        """Test health check."""
        response = test_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_version" in data
        assert "uptime_seconds" in data


class TestInfoEndpoint:
    """Tests for /info endpoint."""

    def test_get_info(self, test_client):
        """Test getting model info."""
        response = test_client.get("/info")

        assert response.status_code == 200
        data = response.json()
        assert "model_version" in data
        assert "n_clusters" in data
        assert "features" in data
        assert data["n_clusters"] == 5


class TestMetricsEndpoint:
    """Tests for /metrics endpoint."""

    def test_get_metrics(self, test_client):
        """Test getting Prometheus metrics."""
        response = test_client.get("/metrics")

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; charset=utf-8"
        assert "http_requests_total" in response.text


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root(self, test_client):
        """Test root endpoint."""
        response = test_client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "docs_url" in data
