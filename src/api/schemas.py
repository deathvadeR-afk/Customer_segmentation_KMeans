"""Pydantic models for API request/response validation."""
from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class CustomerInput(BaseModel):
    """Input schema for single customer prediction."""

    annual_income: float = Field(
        ge=15,
        le=137,
        description="Annual income in thousands of dollars",
        examples=[60.0],
    )
    spending_score: float = Field(
        ge=1,
        le=99,
        description="Spending score (1-100)",
        examples=[50.0],
    )


class PredictionResponse(BaseModel):
    """Response schema for prediction."""

    cluster_id: int = Field(ge=0, le=4, description="Predicted cluster ID (0-4)")
    cluster_name: str = Field(description="Descriptive cluster name")
    model_version: str = Field(description="Model version used for prediction")
    request_id: str = Field(description="Unique request identifier")
    timestamp: datetime = Field(description="Prediction timestamp")


class BatchPredictionRequest(BaseModel):
    """Request schema for batch predictions."""

    customers: List[CustomerInput] = Field(
        max_length=100,
        description="List of customers to predict (max 100)",
    )


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""

    predictions: List[PredictionResponse] = Field(
        description="List of predictions for each customer"
    )
    total_count: int = Field(description="Total number of predictions")
    request_id: str = Field(description="Unique request identifier")
    timestamp: datetime = Field(description="Batch prediction timestamp")


class HealthResponse(BaseModel):
    """Response schema for health check."""

    status: Literal["healthy", "unhealthy"] = Field(description="Service health status")
    model_version: str = Field(description="Currently loaded model version")
    uptime_seconds: float = Field(description="Application uptime in seconds")
    timestamp: datetime = Field(description="Health check timestamp")


class ModelInfoResponse(BaseModel):
    """Response schema for model information."""

    model_version: str = Field(description="Model version")
    training_date: Optional[str] = Field(
        default=None, description="Model training date"
    )
    n_clusters: int = Field(description="Number of clusters")
    features: List[str] = Field(description="Feature names used for clustering")
    performance_metrics: Optional[dict] = Field(
        default=None, description="Model performance metrics"
    )


class RetrainRequest(BaseModel):
    """Request schema for model retraining."""

    n_clusters: int = Field(
        default=5,
        ge=2,
        le=10,
        description="Number of clusters to train",
    )
    data_path: str = Field(
        default="data/Mall_Customers.csv",
        description="Path to training data",
    )


class RetrainResponse(BaseModel):
    """Response schema for retraining request."""

    job_id: str = Field(description="Unique job identifier")
    status: Literal["started"] = Field(description="Job status")
    message: str = Field(description="Status message")
    timestamp: datetime = Field(description="Job start timestamp")


class ErrorResponse(BaseModel):
    """Response schema for errors."""

    error: str = Field(description="Error type")
    detail: str = Field(description="Detailed error message")
    request_id: str = Field(description="Request identifier for tracking")
    timestamp: datetime = Field(description="Error timestamp")
