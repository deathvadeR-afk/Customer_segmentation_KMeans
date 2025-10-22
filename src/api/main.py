"""FastAPI application for customer segmentation."""
import asyncio
import time
import uuid
from datetime import datetime
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import generate_latest

from src.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerInput,
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    RetrainRequest,
    RetrainResponse,
)
from src.models.predict import ModelPredictor
from src.models.train import train_model
from src.monitoring import metrics
from src.utils.config import get_settings
from src.utils.logger import get_logger, setup_logging

# Initialize settings and logging
config = get_settings()
setup_logging(config.log_level, config.environment)
logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title=config.app_name,
    version=config.version,
    description="Production-grade ML API for customer segmentation using K-Means clustering",
)

# Global state
predictor: Optional[ModelPredictor] = None
start_time: float = None


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    global predictor, start_time

    logger.info("=" * 60)
    logger.info(f"Starting {config.app_name} v{config.version}")
    logger.info(f"Environment: {config.environment}")
    logger.info("=" * 60)

    start_time = time.time()

    # Initialize metrics
    metrics.init_metrics()
    logger.info("Metrics initialized")

    # Initialize predictor
    try:
        predictor = ModelPredictor(config)
        metrics.set_model_info(
            version=predictor.version,
            n_clusters=config.n_clusters,
            training_date=predictor.load_time.isoformat() if predictor.load_time else "unknown",
        )
        logger.info(f"Model loaded: version={predictor.version}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    # Start background task for system metrics
    asyncio.create_task(update_system_metrics_task())

    logger.info("Application started successfully")
    logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown."""
    logger.info("Application shutting down")


async def update_system_metrics_task():
    """Background task to update system metrics periodically."""
    while True:
        try:
            uptime = time.time() - start_time
            metrics.app_uptime_seconds.set(uptime)
            metrics.update_system_metrics()
        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")
        await asyncio.sleep(60)  # Update every 60 seconds


@app.middleware("http")
async def add_request_id_and_timing(request: Request, call_next):
    """Middleware to add request ID and track timing."""
    # Generate request ID
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    # Track request
    endpoint = request.url.path
    method = request.method

    # Increment in-progress counter
    if config.metrics_enabled:
        metrics.http_requests_in_progress.labels(endpoint=endpoint).inc()

    # Time the request
    start = time.time()

    # Process request
    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception as e:
        logger.error(f"Request failed: {e}", extra={"request_id": request_id})
        status_code = 500
        raise
    finally:
        duration = time.time() - start

        # Record metrics
        if config.metrics_enabled:
            metrics.record_request(endpoint, method, status_code, duration)
            metrics.http_requests_in_progress.labels(endpoint=endpoint).dec()

    # Add request ID to response headers
    response.headers["X-Request-ID"] = request_id

    return response


# Add CORS middleware if enabled
if config.cors_enabled:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": config.app_name,
        "version": config.version,
        "docs_url": "/docs",
        "health_url": "/health",
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(customer: CustomerInput, request: Request):
    """
    Predict customer segment for a single customer.

    Args:
        customer: Customer input with annual_income and spending_score
        request: FastAPI request object

    Returns:
        PredictionResponse with cluster_id, cluster_name, etc.
    """
    request_id = request.state.request_id

    try:
        # Start timing
        start = time.time()

        # Make prediction
        result = predictor.predict_single(
            customer.annual_income,
            customer.spending_score,
        )

        # Calculate duration
        duration = time.time() - start

        # Record metrics
        if config.metrics_enabled:
            metrics.record_prediction(result["cluster_id"], duration)

        # Build response
        response = PredictionResponse(
            cluster_id=result["cluster_id"],
            cluster_name=result["cluster_name"],
            model_version=result["model_version"],
            request_id=request_id,
            timestamp=datetime.now(),
        )

        return response

    except ValueError as e:
        # Validation error
        if config.metrics_enabled:
            metrics.record_validation_error()
        logger.warning(f"Validation error: {e}", extra={"request_id": request_id})
        raise HTTPException(status_code=422, detail=str(e))

    except Exception as e:
        # Server error
        if config.metrics_enabled:
            metrics.record_error("server")
        logger.error(f"Prediction error: {e}", extra={"request_id": request_id})
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(batch_request: BatchPredictionRequest, request: Request):
    """
    Predict customer segments for multiple customers.

    Args:
        batch_request: Batch prediction request with list of customers
        request: FastAPI request object

    Returns:
        BatchPredictionResponse with list of predictions
    """
    request_id = request.state.request_id

    try:
        # Start timing
        start = time.time()

        # Convert to list of dicts
        customers = [
            {
                "annual_income": c.annual_income,
                "spending_score": c.spending_score,
            }
            for c in batch_request.customers
        ]

        # Make batch prediction
        results = predictor.predict_batch(customers)

        # Calculate duration
        duration = time.time() - start

        # Record metrics
        if config.metrics_enabled:
            metrics.record_batch_prediction(len(customers), duration)

        # Build response
        predictions = [
            PredictionResponse(
                cluster_id=r["cluster_id"],
                cluster_name=r["cluster_name"],
                model_version=r["model_version"],
                request_id=request_id,
                timestamp=datetime.now(),
            )
            for r in results
        ]

        response = BatchPredictionResponse(
            predictions=predictions,
            total_count=len(predictions),
            request_id=request_id,
            timestamp=datetime.now(),
        )

        return response

    except ValueError as e:
        # Validation error
        if config.metrics_enabled:
            metrics.record_validation_error()
        logger.warning(f"Batch validation error: {e}", extra={"request_id": request_id})
        raise HTTPException(status_code=422, detail=str(e))

    except Exception as e:
        # Server error
        if config.metrics_enabled:
            metrics.record_error("server")
        logger.error(f"Batch prediction error: {e}", extra={"request_id": request_id})
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/retrain", response_model=RetrainResponse)
async def retrain(
    background_tasks: BackgroundTasks,
    retrain_request: RetrainRequest,
    request: Request,
    x_api_key: str = Header(..., alias="X-API-Key"),
):
    """
    Trigger model retraining (requires API key).

    Args:
        background_tasks: FastAPI background tasks
        retrain_request: Retraining parameters
        request: FastAPI request object
        x_api_key: API key for authentication

    Returns:
        RetrainResponse with job_id and status
    """
    request_id = request.state.request_id

    # Validate API key
    if x_api_key != config.api_key:
        if config.metrics_enabled:
            metrics.record_error("auth")
        logger.warning("Invalid API key", extra={"request_id": request_id})
        raise HTTPException(status_code=401, detail="Invalid API key")

    # Generate job ID
    job_id = str(uuid.uuid4())

    # Add retraining task to background
    background_tasks.add_task(
        retrain_task,
        job_id,
        retrain_request.n_clusters,
        retrain_request.data_path,
    )

    # Increment retraining counter
    if config.metrics_enabled:
        metrics.retraining_total.inc()

    logger.info(f"Retraining job started: {job_id}", extra={"request_id": request_id})

    response = RetrainResponse(
        job_id=job_id,
        status="started",
        message=f"Retraining job {job_id} started in background",
        timestamp=datetime.now(),
    )

    return response


async def retrain_task(job_id: str, n_clusters: int, data_path: str):
    """
    Background task for model retraining.

    Args:
        job_id: Unique job identifier
        n_clusters: Number of clusters
        data_path: Path to training data
    """
    global predictor

    logger.info(f"Retraining task started: {job_id}")
    start = time.time()

    try:
        # Update config with new n_clusters
        config.n_clusters = n_clusters

        # Train model
        model, train_metrics = train_model(data_path, config)

        # Calculate duration
        duration = time.time() - start

        # Record metrics
        if config.metrics_enabled:
            metrics.retraining_duration_seconds.observe(duration)

        # Reload predictor
        predictor.load_model()

        # Update model info metric
        metrics.set_model_info(
            version=predictor.version,
            n_clusters=n_clusters,
            training_date=datetime.now().isoformat(),
        )

        logger.info(
            f"Retraining completed: {job_id} in {duration:.2f}s, "
            f"silhouette={train_metrics['silhouette_score']:.4f}"
        )

    except Exception as e:
        logger.error(f"Retraining failed: {job_id}, error: {e}")
        if config.metrics_enabled:
            metrics.record_error("training")


@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Health check endpoint.

    Returns:
        HealthResponse with status and uptime
    """
    uptime = time.time() - start_time

    response = HealthResponse(
        status="healthy",
        model_version=predictor.version,
        uptime_seconds=uptime,
        timestamp=datetime.now(),
    )

    return response


@app.get("/metrics")
async def get_metrics():
    """
    Prometheus metrics endpoint.

    Returns:
        Prometheus metrics in text format
    """
    metrics_output = generate_latest()
    return Response(content=metrics_output, media_type="text/plain")


@app.get("/info", response_model=ModelInfoResponse)
async def info():
    """
    Get model information.

    Returns:
        ModelInfoResponse with model metadata
    """
    model_info = predictor.get_model_info()

    response = ModelInfoResponse(
        model_version=model_info["model_version"],
        training_date=model_info.get("loaded_at"),
        n_clusters=model_info["n_clusters"],
        features=model_info["features"],
        performance_metrics=None,  # Could add silhouette score here
    )

    return response


# Exception handlers
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle ValueError (validation errors)."""
    request_id = getattr(request.state, "request_id", "unknown")
    return ErrorResponse(
        error="ValidationError",
        detail=str(exc),
        request_id=request_id,
        timestamp=datetime.now(),
    )


@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(request: Request, exc: FileNotFoundError):
    """Handle FileNotFoundError."""
    request_id = getattr(request.state, "request_id", "unknown")
    return ErrorResponse(
        error="FileNotFoundError",
        detail=str(exc),
        request_id=request_id,
        timestamp=datetime.now(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error(f"Unhandled exception: {exc}", extra={"request_id": request_id})
    return ErrorResponse(
        error="InternalServerError",
        detail="An internal server error occurred",
        request_id=request_id,
        timestamp=datetime.now(),
    )
