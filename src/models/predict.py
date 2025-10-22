"""Model inference module for predictions."""
import os
from datetime import datetime
from typing import Dict, List

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.cluster import KMeans

from src.data.loader import validate_customer_input
from src.utils.config import Settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelPredictor:
    """Model predictor for customer segmentation."""

    def __init__(self, config: Settings):
        """
        Initialize predictor.

        Args:
            config: Settings instance
        """
        self.config = config
        self.logger = get_logger(__name__)
        self.model: KMeans = None
        self.load_time: datetime = None
        self.version: str = "unknown"
        self.load_model()

    def load_model(self) -> None:
        """
        Load model from disk or MLflow registry.

        Raises:
            FileNotFoundError: If model cannot be loaded
        """
        self.logger.info(f"Loading model from {self.config.model_path}")

        # Try loading from local path first
        if os.path.exists(self.config.model_path):
            try:
                self.model = joblib.load(self.config.model_path)
                self.load_time = datetime.now()
                # Get version from file modification time
                mod_time = os.path.getmtime(self.config.model_path)
                self.version = datetime.fromtimestamp(mod_time).strftime(
                    "%Y%m%d_%H%M%S"
                )
                self.logger.info(
                    f"Model loaded successfully from local file (version: {self.version})"
                )
                return
            except Exception as e:
                self.logger.error(f"Error loading model from file: {e}")

        # Try loading from MLflow registry
        try:
            self.logger.info(
                f"Attempting to load model from MLflow registry: {self.config.model_registry_name}"
            )
            mlflow.set_tracking_uri(self.config.tracking_uri)

            # Load production model from registry
            model_uri = f"models:/{self.config.model_registry_name}/Production"
            self.model = mlflow.sklearn.load_model(model_uri)
            self.load_time = datetime.now()
            self.version = "production"

            # Save to local path for faster future loads
            os.makedirs(os.path.dirname(self.config.model_path), exist_ok=True)
            joblib.dump(self.model, self.config.model_path)

            self.logger.info(
                f"Model loaded from MLflow registry and saved to {self.config.model_path}"
            )
            return
        except Exception as e:
            self.logger.error(f"Error loading model from MLflow: {e}")

        # If we get here, model couldn't be loaded
        raise FileNotFoundError(
            f"Could not load model from {self.config.model_path} or MLflow registry. "
            "Please train a model first using: python src/models/train.py"
        )

    def predict_single(self, annual_income: float, spending_score: float) -> Dict:
        """
        Predict cluster for a single customer.

        Args:
            annual_income: Annual income in thousands
            spending_score: Spending score (1-100)

        Returns:
            Dict with cluster_id, cluster_name, model_version, prediction_timestamp

        Raises:
            ValueError: If input values are invalid
        """
        # Validate inputs
        validate_customer_input(annual_income, spending_score, self.config)

        # Create input array
        X = np.array([[annual_income, spending_score]])

        # Make prediction
        cluster_id = int(self.model.predict(X)[0])

        # Get cluster name
        cluster_name = self.get_cluster_name(cluster_id)

        result = {
            "cluster_id": cluster_id,
            "cluster_name": cluster_name,
            "model_version": self.version,
            "prediction_timestamp": datetime.now().isoformat(),
        }

        self.logger.debug(
            f"Prediction: income={annual_income}, score={spending_score} -> cluster={cluster_id}"
        )

        return result

    def predict_batch(self, customers: List[Dict]) -> List[Dict]:
        """
        Predict clusters for multiple customers.

        Args:
            customers: List of customer dicts with annual_income and spending_score

        Returns:
            List of prediction dicts

        Raises:
            ValueError: If batch size exceeds limit or inputs are invalid
        """
        if len(customers) > self.config.batch_max_size:
            raise ValueError(
                f"Batch size {len(customers)} exceeds maximum {self.config.batch_max_size}"
            )

        # Validate all inputs first
        for i, customer in enumerate(customers):
            try:
                validate_customer_input(
                    customer["annual_income"],
                    customer["spending_score"],
                    self.config,
                )
            except ValueError as e:
                raise ValueError(f"Invalid input at index {i}: {str(e)}")

        # Create input array
        X = np.array(
            [[c["annual_income"], c["spending_score"]] for c in customers]
        )

        # Make predictions
        cluster_ids = self.model.predict(X)

        # Build results
        results = []
        for i, cluster_id in enumerate(cluster_ids):
            result = {
                "cluster_id": int(cluster_id),
                "cluster_name": self.get_cluster_name(int(cluster_id)),
                "model_version": self.version,
                "prediction_timestamp": datetime.now().isoformat(),
            }
            results.append(result)

        self.logger.info(f"Batch prediction completed for {len(customers)} customers")

        return results

    def get_cluster_name(self, cluster_id: int) -> str:
        """
        Map cluster ID to descriptive name.

        Args:
            cluster_id: Cluster ID (0-4)

        Returns:
            str: Descriptive cluster name
        """
        # Note: These are generic names. In production, you would analyze
        # the actual centroids to determine appropriate names.
        cluster_names = {
            0: "High Income, High Spending",
            1: "High Income, Low Spending",
            2: "Low Income, Low Spending",
            3: "Low Income, High Spending",
            4: "Medium Income, Medium Spending",
        }
        return cluster_names.get(cluster_id, f"Cluster {cluster_id}")

    def get_model_info(self) -> Dict:
        """
        Get model information.

        Returns:
            Dict with model metadata
        """
        info = {
            "model_version": self.version,
            "n_clusters": self.config.n_clusters,
            "features": self.config.feature_columns,
            "loaded_at": self.load_time.isoformat() if self.load_time else None,
            "model_path": self.config.model_path,
        }

        # Add centroid information if available
        if self.model and hasattr(self.model, "cluster_centers_"):
            info["centroids"] = self.model.cluster_centers_.tolist()

        return info
