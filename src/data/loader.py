"""Data loading and preprocessing module."""
import numpy as np
import pandas as pd

from src.utils.config import Settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load customer data from CSV file.

    Args:
        file_path: Path to CSV file

    Returns:
        pd.DataFrame: Loaded customer data

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If data is invalid
    """
    logger.info(f"Loading data from {file_path}")

    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        logger.error(f"Data file not found: {file_path}")
        raise FileNotFoundError(f"Data file not found: {file_path}")
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise ValueError(f"Error loading CSV file: {str(e)}")

    # Validate required columns
    required_columns = [
        "CustomerID",
        "Gender",
        "Age",
        "Annual Income (k$)",
        "Spending Score (1-100)",
    ]
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Check for missing values
    if df.isnull().any().any():
        null_counts = df.isnull().sum()
        logger.error(f"Data contains missing values: {null_counts[null_counts > 0]}")
        raise ValueError("Data contains missing values")

    # Check for duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        logger.warning(f"Found {duplicates} duplicate rows, keeping first occurrence")
        df = df.drop_duplicates(keep="first")

    logger.info(f"Successfully loaded {len(df)} customer records")
    return df


def extract_features(df: pd.DataFrame) -> np.ndarray:
    """
    Extract features for clustering (Annual Income and Spending Score).

    Args:
        df: Customer DataFrame

    Returns:
        np.ndarray: Feature array with shape (n_samples, 2)

    Raises:
        ValueError: If required columns are missing
    """
    logger.info("Extracting features for clustering")

    required_columns = ["Annual Income (k$)", "Spending Score (1-100)"]
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Extract columns 3 and 4 (Annual Income and Spending Score)
    X = df.iloc[:, 3:5].values

    # Validate shape
    if X.shape[1] != 2:
        raise ValueError(f"Expected 2 features, got {X.shape[1]}")

    logger.info(f"Extracted features with shape {X.shape}")
    return X


def validate_customer_input(
    annual_income: float, spending_score: float, config: Settings
) -> None:
    """
    Validate customer input values.

    Args:
        annual_income: Annual income in thousands
        spending_score: Spending score (1-100)
        config: Settings instance with validation ranges

    Raises:
        ValueError: If input values are out of range
    """
    if not (config.income_min <= annual_income <= config.income_max):
        raise ValueError(
            f"Annual income must be between {config.income_min} and {config.income_max}, got {annual_income}"
        )

    if not (config.score_min <= spending_score <= config.score_max):
        raise ValueError(
            f"Spending score must be between {config.score_min} and {config.score_max}, got {spending_score}"
        )

    logger.debug(
        f"Input validation passed: income={annual_income}, score={spending_score}"
    )
