"""Pytest fixtures for testing."""
import os
import tempfile
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.cluster import KMeans

from src.utils.config import Settings


@pytest.fixture(scope="session")
def test_config():
    """Create test configuration."""
    config = Settings(
        app_name="Test Customer Segmentation API",
        version="test",
        environment="test",
        debug=True,
        log_level="DEBUG",
        model_path="test_models/test_model.pkl",
        n_clusters=5,
        random_state=0,
        data_path="test_data/test_customers.csv",
        batch_max_size=10,
        api_key="test_api_key",
        tracking_uri="file:./test_mlruns",
        experiment_name="test_experiment",
        model_registry_name="test_model",
    )
    return config


@pytest.fixture(scope="session")
def test_data():
    """Create test customer data."""
    data = {
        "CustomerID": range(1, 11),
        "Gender": ["Male", "Female"] * 5,
        "Age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        "Annual Income (k$)": [15, 20, 40, 60, 80, 100, 120, 137, 50, 70],
        "Spending Score (1-100)": [1, 10, 30, 50, 70, 90, 99, 80, 40, 60],
    }
    df = pd.DataFrame(data)
    return df


@pytest.fixture(scope="session")
def test_model(test_data):
    """Create and train a test model."""
    # Extract features
    X = test_data.iloc[:, 3:5].values

    # Train simple model
    model = KMeans(n_clusters=5, init="k-means++", random_state=0, n_init=10)
    model.fit(X)

    # Save to temporary file
    temp_dir = tempfile.mkdtemp()
    model_path = os.path.join(temp_dir, "test_model.pkl")

    import joblib
    joblib.dump(model, model_path)

    yield model, model_path

    # Cleanup
    try:
        os.remove(model_path)
        os.rmdir(temp_dir)
    except Exception:
        pass


@pytest.fixture(scope="session")
def test_data_file(test_data):
    """Create temporary CSV file with test data."""
    temp_dir = tempfile.mkdtemp()
    csv_path = os.path.join(temp_dir, "test_customers.csv")
    test_data.to_csv(csv_path, index=False)

    yield csv_path

    # Cleanup
    try:
        os.remove(csv_path)
        os.rmdir(temp_dir)
    except Exception:
        pass


@pytest.fixture
def mock_mlflow():
    """Mock MLflow tracking functions."""
    with patch("mlflow.start_run") as mock_start, \
         patch("mlflow.log_params") as mock_params, \
         patch("mlflow.log_metric") as mock_metric, \
         patch("mlflow.log_artifact") as mock_artifact, \
         patch("mlflow.sklearn.log_model") as mock_log_model, \
         patch("mlflow.set_tracking_uri") as mock_uri, \
         patch("mlflow.set_experiment") as mock_exp, \
         patch("mlflow.get_experiment_by_name") as mock_get_exp:

        # Configure mocks
        mock_run = Mock()
        mock_run.info.run_id = "test_run_id"
        mock_start.return_value.__enter__.return_value = mock_run
        mock_get_exp.return_value = None

        yield {
            "start_run": mock_start,
            "log_params": mock_params,
            "log_metric": mock_metric,
            "log_artifact": mock_artifact,
            "log_model": mock_log_model,
            "set_tracking_uri": mock_uri,
            "set_experiment": mock_exp,
            "get_experiment_by_name": mock_get_exp,
        }


@pytest.fixture
def test_client(test_config, test_model):
    """Create FastAPI test client."""
    model, model_path = test_model

    # Patch config and predictor
    with patch("src.api.main.config", test_config), \
         patch("src.api.main.predictor") as mock_predictor:

        # Configure mock predictor
        mock_predictor.version = "test_version"
        mock_predictor.config = test_config
        mock_predictor.model = model

        # Mock predict methods
        def mock_predict_single(income, score):
            return {
                "cluster_id": 0,
                "cluster_name": "Test Cluster",
                "model_version": "test_version",
                "prediction_timestamp": "2024-01-01T00:00:00",
            }

        def mock_predict_batch(customers):
            return [mock_predict_single(c["annual_income"], c["spending_score"]) for c in customers]

        def mock_get_model_info():
            return {
                "model_version": "test_version",
                "n_clusters": 5,
                "features": ["Annual Income (k$)", "Spending Score (1-100)"],
                "loaded_at": "2024-01-01T00:00:00",
                "model_path": model_path,
            }

        mock_predictor.predict_single = Mock(side_effect=mock_predict_single)
        mock_predictor.predict_batch = Mock(side_effect=mock_predict_batch)
        mock_predictor.get_model_info = Mock(side_effect=mock_get_model_info)

        # Import app after patching
        from src.api.main import app

        # Disable startup/shutdown events for testing
        app.router.on_startup = []
        app.router.on_shutdown = []

        client = TestClient(app)
        yield client
