"""Model training module with MLflow tracking."""
import json
import os
import time
from datetime import datetime
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.data.loader import extract_features, load_data
from src.utils.config import Settings, get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def save_elbow_plot(wcss_values: list, output_path: str) -> None:
    """
    Create and save elbow method plot.

    Args:
        wcss_values: List of WCSS values for different k
        output_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    k_values = range(1, len(wcss_values) + 1)
    plt.plot(k_values, wcss_values, "bo-", linewidth=2, markersize=8)
    plt.xlabel("Number of Clusters (k)", fontsize=12)
    plt.ylabel("WCSS (Within-Cluster Sum of Squares)", fontsize=12)
    plt.title("Elbow Method For Optimal k", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Elbow plot saved to {output_path}")


def save_cluster_plot(
    X: np.ndarray, labels: np.ndarray, centroids: np.ndarray, output_path: str
) -> None:
    """
    Create and save cluster visualization.

    Args:
        X: Feature array (n_samples, 2)
        labels: Cluster labels
        centroids: Cluster centroids
        output_path: Path to save the plot
    """
    plt.figure(figsize=(12, 8))

    # Define colors for clusters
    colors = ["green", "red", "yellow", "violet", "blue"]
    cluster_names = [
        "Cluster 0",
        "Cluster 1",
        "Cluster 2",
        "Cluster 3",
        "Cluster 4",
    ]

    # Plot each cluster
    for i in range(len(centroids)):
        cluster_points = X[labels == i]
        plt.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            s=100,
            c=colors[i],
            label=cluster_names[i],
            alpha=0.6,
            edgecolors="black",
            linewidth=0.5,
        )

    # Plot centroids
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        s=300,
        c="cyan",
        label="Centroids",
        alpha=1,
        edgecolors="black",
        linewidth=2,
        marker="*",
    )

    plt.xlabel("Annual Income (k$)", fontsize=12)
    plt.ylabel("Spending Score (1-100)", fontsize=12)
    plt.title("Customer Segments", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10, loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Cluster plot saved to {output_path}")


def train_model(data_path: str, config: Settings) -> Tuple[KMeans, Dict]:
    """
    Train K-Means model with MLflow tracking.

    Args:
        data_path: Path to training data CSV
        config: Settings instance

    Returns:
        Tuple of (trained model, metrics dict)
    """
    logger.info("Starting model training")
    start_time = time.time()

    # Set MLflow tracking URI
    mlflow.set_tracking_uri(config.tracking_uri)

    # Set or create experiment
    experiment = mlflow.get_experiment_by_name(config.experiment_name)
    if experiment is None:
        mlflow.create_experiment(config.experiment_name)
    mlflow.set_experiment(config.experiment_name)

    # Load and prepare data
    df = load_data(data_path)
    X = extract_features(df)

    # Start MLflow run
    with mlflow.start_run() as run:
        logger.info(f"MLflow run started: {run.info.run_id}")

        # Log parameters
        params = {
            "n_clusters": config.n_clusters,
            "init_method": config.init_method,
            "random_state": config.random_state,
            "max_iter": config.max_iter,
            "n_init": 10,
        }
        mlflow.log_params(params)

        # Calculate WCSS for elbow method (k=1 to 10)
        logger.info("Calculating WCSS for elbow method")
        wcss_values = []
        for k in range(1, 11):
            kmeans_temp = KMeans(
                n_clusters=k,
                init=config.init_method,
                random_state=config.random_state,
                n_init=10,
            )
            kmeans_temp.fit(X)
            wcss_values.append(kmeans_temp.inertia_)
            mlflow.log_metric(f"wcss_{k}", kmeans_temp.inertia_)

        # Train final model
        logger.info(f"Training final model with {config.n_clusters} clusters")
        model = KMeans(
            n_clusters=config.n_clusters,
            init=config.init_method,
            random_state=config.random_state,
            max_iter=config.max_iter,
            n_init=10,
        )
        labels = model.fit_predict(X)

        # Calculate metrics
        silhouette = silhouette_score(X, labels)
        training_time = time.time() - start_time
        n_samples = len(X)

        metrics = {
            "silhouette_score": silhouette,
            "training_time_seconds": training_time,
            "n_samples": n_samples,
            "n_clusters": config.n_clusters,
            "final_wcss": model.inertia_,
        }

        # Log metrics
        mlflow.log_metric("silhouette_score", silhouette)
        mlflow.log_metric("training_time_seconds", training_time)
        mlflow.log_metric("n_samples", n_samples)
        mlflow.log_metric("final_wcss", model.inertia_)

        logger.info(f"Model trained - Silhouette Score: {silhouette:.4f}")

        # Create and log artifacts
        os.makedirs("temp_artifacts", exist_ok=True)

        # Save elbow plot
        elbow_path = "temp_artifacts/elbow_method.png"
        save_elbow_plot(wcss_values, elbow_path)
        mlflow.log_artifact(elbow_path)

        # Save cluster visualization
        cluster_path = "temp_artifacts/clusters.png"
        save_cluster_plot(X, labels, model.cluster_centers_, cluster_path)
        mlflow.log_artifact(cluster_path)

        # Save training data sample
        sample_path = "temp_artifacts/train_data_sample.csv"
        df.head(20).to_csv(sample_path, index=False)
        mlflow.log_artifact(sample_path)

        # Save model metadata
        metadata = {
            "model_type": "KMeans",
            "n_clusters": config.n_clusters,
            "training_date": datetime.now().isoformat(),
            "n_samples": n_samples,
            "features": config.feature_columns,
            "metrics": metrics,
        }
        metadata_path = "temp_artifacts/model_info.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        mlflow.log_artifact(metadata_path)

        # Log model to MLflow
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name=config.model_registry_name,
        )

        # Save model locally
        os.makedirs(os.path.dirname(config.model_path), exist_ok=True)
        joblib.dump(model, config.model_path)
        logger.info(f"Model saved to {config.model_path}")

        # Transition model to Production stage
        try:
            client = mlflow.tracking.MlflowClient()
            # Get latest model version
            latest_versions = client.get_latest_versions(
                config.model_registry_name, stages=["None"]
            )
            if latest_versions:
                latest_version = latest_versions[0].version
                client.transition_model_version_stage(
                    name=config.model_registry_name,
                    version=latest_version,
                    stage="Production",
                )
                logger.info(
                    f"Model version {latest_version} transitioned to Production"
                )
        except Exception as e:
            logger.warning(f"Could not transition model to Production: {e}")

        # Clean up temp artifacts
        import shutil

        shutil.rmtree("temp_artifacts", ignore_errors=True)

        logger.info(f"Training completed in {training_time:.2f} seconds")
        return model, metrics


if __name__ == "__main__":
    """Run training when script is executed directly."""
    config = get_settings()

    print("=" * 60)
    print("Customer Segmentation Model Training")
    print("=" * 60)

    model, metrics = train_model(config.data_path, config)

    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"Model saved to: {config.model_path}")
    print(f"Silhouette Score: {metrics['silhouette_score']:.4f}")
    print(f"Training Time: {metrics['training_time_seconds']:.2f} seconds")
    print(f"Number of Samples: {metrics['n_samples']}")
    print(f"Number of Clusters: {metrics['n_clusters']}")
    print(f"Final WCSS: {metrics['final_wcss']:.2f}")
    print("=" * 60)
