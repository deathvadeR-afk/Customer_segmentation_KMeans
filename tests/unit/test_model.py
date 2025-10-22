"""Unit tests for model training and prediction."""
import os
import tempfile

import joblib
import numpy as np
import pytest
from sklearn.cluster import KMeans

from src.models.predict import ModelPredictor


class TestModelPredictor:
    """Tests for ModelPredictor class."""

    def test_predictor_initialization(self, test_config, test_model):
        """Test predictor initializes successfully."""
        model, model_path = test_model
        test_config.model_path = model_path

        predictor = ModelPredictor(test_config)

        assert predictor.model is not None
        assert predictor.version is not None
        assert predictor.load_time is not None

    def test_predict_single(self, test_config, test_model):
        """Test single prediction."""
        model, model_path = test_model
        test_config.model_path = model_path

        predictor = ModelPredictor(test_config)
        result = predictor.predict_single(60.0, 50.0)

        assert "cluster_id" in result
        assert "cluster_name" in result
        assert "model_version" in result
        assert 0 <= result["cluster_id"] <= 4

    def test_predict_single_invalid_input(self, test_config, test_model):
        """Test single prediction with invalid input."""
        model, model_path = test_model
        test_config.model_path = model_path

        predictor = ModelPredictor(test_config)

        with pytest.raises(ValueError):
            predictor.predict_single(200.0, 50.0)  # Income too high

    def test_predict_batch(self, test_config, test_model):
        """Test batch prediction."""
        model, model_path = test_model
        test_config.model_path = model_path

        predictor = ModelPredictor(test_config)

        customers = [
            {"annual_income": 60.0, "spending_score": 50.0},
            {"annual_income": 80.0, "spending_score": 70.0},
            {"annual_income": 40.0, "spending_score": 30.0},
        ]

        results = predictor.predict_batch(customers)

        assert len(results) == 3
        for result in results:
            assert "cluster_id" in result
            assert 0 <= result["cluster_id"] <= 4

    def test_predict_batch_exceeds_limit(self, test_config, test_model):
        """Test batch prediction exceeds size limit."""
        model, model_path = test_model
        test_config.model_path = model_path
        test_config.batch_max_size = 5

        predictor = ModelPredictor(test_config)

        # Create batch larger than limit
        customers = [
            {"annual_income": 60.0, "spending_score": 50.0}
            for _ in range(10)
        ]

        with pytest.raises(ValueError, match="exceeds maximum"):
            predictor.predict_batch(customers)

    def test_get_cluster_name(self, test_config, test_model):
        """Test cluster name mapping."""
        model, model_path = test_model
        test_config.model_path = model_path

        predictor = ModelPredictor(test_config)

        for cluster_id in range(5):
            name = predictor.get_cluster_name(cluster_id)
            assert isinstance(name, str)
            assert len(name) > 0

    def test_get_model_info(self, test_config, test_model):
        """Test getting model information."""
        model, model_path = test_model
        test_config.model_path = model_path

        predictor = ModelPredictor(test_config)
        info = predictor.get_model_info()

        assert "model_version" in info
        assert "n_clusters" in info
        assert "features" in info
        assert info["n_clusters"] == 5

    def test_model_not_found(self, test_config):
        """Test handling missing model file."""
        test_config.model_path = "nonexistent_model.pkl"
        test_config.tracking_uri = "file:./nonexistent_mlruns"

        with pytest.raises(FileNotFoundError):
            ModelPredictor(test_config)
