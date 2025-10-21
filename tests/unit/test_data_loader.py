"""Unit tests for data loading module."""
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.data.loader import extract_features, load_data, validate_customer_input


class TestLoadData:
    """Tests for load_data function."""

    def test_load_data_success(self, test_data_file):
        """Test loading CSV file successfully."""
        df = load_data(test_data_file)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10
        assert "Annual Income (k$)" in df.columns
        assert "Spending Score (1-100)" in df.columns

    def test_load_data_missing_file(self):
        """Test handling missing file."""
        with pytest.raises(FileNotFoundError):
            load_data("nonexistent_file.csv")

    def test_load_data_missing_columns(self):
        """Test handling missing columns."""
        # Create temp file with missing columns
        temp_dir = tempfile.mkdtemp()
        csv_path = os.path.join(temp_dir, "invalid.csv")

        invalid_df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        invalid_df.to_csv(csv_path, index=False)

        try:
            with pytest.raises(ValueError, match="Missing required columns"):
                load_data(csv_path)
        finally:
            os.remove(csv_path)
            os.rmdir(temp_dir)

    def test_load_data_with_missing_values(self):
        """Test handling missing values."""
        temp_dir = tempfile.mkdtemp()
        csv_path = os.path.join(temp_dir, "missing_values.csv")

        df_with_nulls = pd.DataFrame({
            "CustomerID": [1, 2, 3],
            "Gender": ["Male", None, "Female"],
            "Age": [25, 30, 35],
            "Annual Income (k$)": [50, 60, 70],
            "Spending Score (1-100)": [50, 60, 70],
        })
        df_with_nulls.to_csv(csv_path, index=False)

        try:
            with pytest.raises(ValueError, match="missing values"):
                load_data(csv_path)
        finally:
            os.remove(csv_path)
            os.rmdir(temp_dir)


class TestExtractFeatures:
    """Tests for extract_features function."""

    def test_extract_features_success(self, test_data):
        """Test feature extraction."""
        X = extract_features(test_data)
        assert isinstance(X, np.ndarray)
        assert X.shape == (10, 2)
        assert X.shape[1] == 2  # Two features

    def test_extract_features_missing_columns(self):
        """Test handling missing feature columns."""
        invalid_df = pd.DataFrame({
            "CustomerID": [1, 2],
            "Gender": ["Male", "Female"],
            "Age": [25, 30],
        })

        with pytest.raises(ValueError, match="Missing required columns"):
            extract_features(invalid_df)


class TestValidateCustomerInput:
    """Tests for validate_customer_input function."""

    def test_validate_valid_input(self, test_config):
        """Test validation passes for valid input."""
        # Should not raise
        validate_customer_input(60.0, 50.0, test_config)

    def test_validate_income_too_low(self, test_config):
        """Test validation fails for income too low."""
        with pytest.raises(ValueError, match="Annual income must be between"):
            validate_customer_input(10.0, 50.0, test_config)

    def test_validate_income_too_high(self, test_config):
        """Test validation fails for income too high."""
        with pytest.raises(ValueError, match="Annual income must be between"):
            validate_customer_input(200.0, 50.0, test_config)

    def test_validate_score_too_low(self, test_config):
        """Test validation fails for score too low."""
        with pytest.raises(ValueError, match="Spending score must be between"):
            validate_customer_input(60.0, 0.0, test_config)

    def test_validate_score_too_high(self, test_config):
        """Test validation fails for score too high."""
        with pytest.raises(ValueError, match="Spending score must be between"):
            validate_customer_input(60.0, 100.0, test_config)
