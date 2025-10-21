"""Prometheus metrics definitions and tracking."""
import time

import psutil
from prometheus_client import Counter, Gauge, Histogram, Info

# Request metrics
http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["endpoint", "method", "status_code"],
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["endpoint", "method"],
)

http_requests_in_progress = Gauge(
    "http_requests_in_progress",
    "Number of HTTP requests in progress",
    ["endpoint"],
)

# Prediction metrics
predictions_total = Counter(
    "predictions_total",
    "Total number of predictions made",
    ["cluster_id"],
)

prediction_duration_seconds = Histogram(
    "prediction_duration_seconds",
    "Prediction duration in seconds",
)

batch_predictions_total = Counter(
    "batch_predictions_total",
    "Total number of batch predictions",
)

batch_size = Histogram(
    "batch_size",
    "Size of batch predictions",
)

# Model metrics
model_info = Info(
    "model_info",
    "Information about the loaded model",
)

model_load_time_seconds = Gauge(
    "model_load_time_seconds",
    "Time taken to load the model in seconds",
)

retraining_total = Counter(
    "retraining_total",
    "Total number of retraining requests",
)

retraining_duration_seconds = Histogram(
    "retraining_duration_seconds",
    "Model retraining duration in seconds",
)

# Error metrics
errors_total = Counter(
    "errors_total",
    "Total number of errors",
    ["error_type"],
)

validation_errors_total = Counter(
    "validation_errors_total",
    "Total number of validation errors",
)

# System metrics
app_uptime_seconds = Gauge(
    "app_uptime_seconds",
    "Application uptime in seconds",
)

memory_usage_bytes = Gauge(
    "memory_usage_bytes",
    "Memory usage in bytes",
)

cpu_usage_percent = Gauge(
    "cpu_usage_percent",
    "CPU usage percentage",
)


def init_metrics() -> None:
    """Initialize metrics with default values."""
    # Set initial uptime
    app_uptime_seconds.set(0)

    # Set initial system metrics
    update_system_metrics()


def record_request(endpoint: str, method: str, status_code: int, duration: float) -> None:
    """
    Record HTTP request metrics.

    Args:
        endpoint: API endpoint path
        method: HTTP method (GET, POST, etc.)
        status_code: HTTP status code
        duration: Request duration in seconds
    """
    http_requests_total.labels(
        endpoint=endpoint,
        method=method,
        status_code=status_code,
    ).inc()

    http_request_duration_seconds.labels(
        endpoint=endpoint,
        method=method,
    ).observe(duration)


def record_prediction(cluster_id: int, duration: float) -> None:
    """
    Record single prediction metrics.

    Args:
        cluster_id: Predicted cluster ID
        duration: Prediction duration in seconds
    """
    predictions_total.labels(cluster_id=cluster_id).inc()
    prediction_duration_seconds.observe(duration)


def record_batch_prediction(batch_size_value: int, duration: float) -> None:
    """
    Record batch prediction metrics.

    Args:
        batch_size_value: Number of predictions in batch
        duration: Total prediction duration in seconds
    """
    batch_predictions_total.inc()
    batch_size.observe(batch_size_value)
    prediction_duration_seconds.observe(duration)


def record_error(error_type: str) -> None:
    """
    Record error occurrence.

    Args:
        error_type: Type of error (validation, server, auth, etc.)
    """
    errors_total.labels(error_type=error_type).inc()


def record_validation_error() -> None:
    """Record validation error occurrence."""
    validation_errors_total.inc()
    record_error("validation")


def update_system_metrics() -> None:
    """Update system resource metrics."""
    try:
        # Get memory usage
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_usage_bytes.set(memory_info.rss)

        # Get CPU usage
        cpu_percent = process.cpu_percent(interval=0.1)
        cpu_usage_percent.set(cpu_percent)
    except Exception:
        # Silently fail if system metrics can't be collected
        pass


def set_model_info(version: str, n_clusters: int, training_date: str) -> None:
    """
    Set model information metric.

    Args:
        version: Model version
        n_clusters: Number of clusters
        training_date: Training date string
    """
    model_info.info(
        {
            "version": version,
            "n_clusters": str(n_clusters),
            "training_date": training_date,
        }
    )
