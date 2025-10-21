"""Unit tests for Prometheus metrics."""
import pytest
from prometheus_client import REGISTRY

from src.monitoring import metrics


class TestMetrics:
    """Tests for Prometheus metrics functions."""

    def test_init_metrics(self):
        """Test metrics initialization."""
        metrics.init_metrics()

        # Check that metrics are initialized
        assert metrics.app_uptime_seconds._value.get() == 0

    def test_record_request(self):
        """Test recording HTTP request metrics."""
        initial_value = metrics.http_requests_total.labels(
            endpoint="/test",
            method="GET",
            status_code=200
        )._value.get()

        metrics.record_request("/test", "GET", 200, 0.5)

        new_value = metrics.http_requests_total.labels(
            endpoint="/test",
            method="GET",
            status_code=200
        )._value.get()

        assert new_value > initial_value

    def test_record_prediction(self):
        """Test recording prediction metrics."""
        initial_value = metrics.predictions_total.labels(
            cluster_id=0
        )._value.get()

        metrics.record_prediction(0, 0.1)

        new_value = metrics.predictions_total.labels(
            cluster_id=0
        )._value.get()

        assert new_value > initial_value

    def test_record_batch_prediction(self):
        """Test recording batch prediction metrics."""
        initial_value = metrics.batch_predictions_total._value.get()

        metrics.record_batch_prediction(10, 0.5)

        new_value = metrics.batch_predictions_total._value.get()

        assert new_value > initial_value

    def test_record_error(self):
        """Test recording error metrics."""
        initial_value = metrics.errors_total.labels(
            error_type="test_error"
        )._value.get()

        metrics.record_error("test_error")

        new_value = metrics.errors_total.labels(
            error_type="test_error"
        )._value.get()

        assert new_value > initial_value

    def test_record_validation_error(self):
        """Test recording validation error."""
        initial_validation = metrics.validation_errors_total._value.get()
        initial_errors = metrics.errors_total.labels(
            error_type="validation"
        )._value.get()

        metrics.record_validation_error()

        new_validation = metrics.validation_errors_total._value.get()
        new_errors = metrics.errors_total.labels(
            error_type="validation"
        )._value.get()

        assert new_validation > initial_validation
        assert new_errors > initial_errors

    def test_update_system_metrics(self):
        """Test updating system metrics."""
        # Should not raise errors
        metrics.update_system_metrics()

        # Memory and CPU gauges should have values
        memory = metrics.memory_usage_bytes._value.get()
        cpu = metrics.cpu_usage_percent._value.get()

        assert memory >= 0
        assert cpu >= 0

    def test_set_model_info(self):
        """Test setting model info metric."""
        metrics.set_model_info(
            version="1.0.0",
            n_clusters=5,
            training_date="2024-01-01"
        )

        # Model info should be set
        info = metrics.model_info._value
        assert info is not None
