"""
preprocessing/imputation.py
============================
Advanced Missing Value Analysis and Imputation for InsightForge.

This module provides:
    1. Per-column analysis of missing values with statistics
    2. Intelligent imputation method recommendation
    3. Configurable imputation application (accepts user overrides)

RECOMMENDATION LOGIC:
    - Numeric + skewed (|skew| > 1) or has >5% outliers  → Median
    - Numeric + normally distributed (|skew| ≤ 1)         → Mean
    - High missing ratio (> 30%) + numeric                → KNN Imputer
    - Very high missing (> 50%) + complex patterns        → Iterative Imputer
    - Categorical (any)                                    → Most Frequent
    - Constant value fill requested explicitly             → Constant
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

logger = logging.getLogger(__name__)


# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_column_for_imputation(series: pd.Series) -> Dict:
    """
    Analyze a single column and return imputation statistics + recommendation.

    Args:
        series: A pandas Series (single column from a DataFrame)

    Returns:
        dict with keys:
            column          (str)   : Column name
            dtype           (str)   : Data type
            total_count     (int)   : Total number of rows
            missing_count   (int)   : Number of missing values
            missing_pct     (float) : Percentage of missing values (0–100)
            is_numeric      (bool)  : Whether the column is numeric
            skewness        (float) : Skewness (None for categorical)
            has_outliers    (bool)  : Whether > 5% of values are outliers
            recommended     (str)   : Recommended imputation method
            reason          (str)   : Human-readable explanation
    """
    col_name = series.name
    total = len(series)
    missing_count = int(series.isnull().sum())
    missing_pct = round((missing_count / total) * 100, 2) if total > 0 else 0.0
    is_numeric = pd.api.types.is_numeric_dtype(series)

    skewness = None
    has_outliers = False
    recommended = 'most_frequent'
    reason = ''

    if is_numeric and missing_count < total:
        non_null = series.dropna()
        try:
            skewness = round(float(non_null.skew()), 3)
        except Exception:
            skewness = 0.0

        # Outlier detection via IQR
        q1, q3 = non_null.quantile(0.25), non_null.quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            outlier_mask = (non_null < q1 - 1.5 * iqr) | (non_null > q3 + 1.5 * iqr)
            outlier_pct = outlier_mask.sum() / len(non_null)
            has_outliers = outlier_pct > 0.05

        # Determine recommendation
        if missing_pct > 50:
            recommended = 'iterative'
            reason = (
                f"Very high missingness ({missing_pct}%). "
                "Iterative Imputer models each column as a function of the others for best accuracy."
            )
        elif missing_pct > 30:
            recommended = 'knn'
            reason = (
                f"High missingness ({missing_pct}%). "
                "KNN Imputer uses nearest-neighbor samples which works well when many values are missing."
            )
        elif has_outliers or (skewness is not None and abs(skewness) > 1.0):
            recommended = 'median'
            reason = (
                f"Numeric column with {'outliers detected' if has_outliers else f'high skewness ({skewness})'} — "
                "Median is robust to extreme values and doesn't get pulled by outliers."
            )
        else:
            recommended = 'mean'
            reason = (
                f"Numeric column with low skewness ({skewness}) and no significant outliers — "
                "Mean imputation preserves the column's distribution."
            )
    else:
        # Categorical / non-numeric
        recommended = 'most_frequent'
        reason = (
            "Categorical column — Most Frequent (mode) imputation preserves the "
            "most common category without introducing artificial values."
        )

    return {
        'column': col_name,
        'dtype': str(series.dtype),
        'total_count': total,
        'missing_count': missing_count,
        'missing_pct': missing_pct,
        'is_numeric': is_numeric,
        'skewness': skewness,
        'has_outliers': has_outliers,
        'recommended': recommended,
        'reason': reason,
    }


def analyze_dataframe_for_imputation(df: pd.DataFrame) -> List[Dict]:
    """
    Analyze all columns in a DataFrame that have missing values.

    Args:
        df: Input DataFrame

    Returns:
        List of analysis dicts (one per column with missing values),
        sorted by missing_pct descending.
    """
    results = []
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            analysis = analyze_column_for_imputation(df[col])
            results.append(analysis)

    # Sort by missing percentage (highest first)
    results.sort(key=lambda x: x['missing_pct'], reverse=True)
    return results


def get_missing_value_summary(df: pd.DataFrame) -> Dict:
    """
    Get a high-level summary of missing values in the DataFrame.

    Returns:
        dict with:
            total_cells     (int): Total cells in DataFrame
            missing_cells   (int): Total missing cells
            missing_pct     (float): Overall missing percentage
            columns_affected (int): Number of columns with any missing
            rows_affected   (int): Number of rows with any missing
    """
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = int(df.isnull().sum().sum())
    return {
        'total_cells': total_cells,
        'missing_cells': missing_cells,
        'missing_pct': round((missing_cells / total_cells) * 100, 2) if total_cells > 0 else 0.0,
        'columns_affected': int((df.isnull().sum() > 0).sum()),
        'rows_affected': int(df.isnull().any(axis=1).sum()),
    }


# ============================================================================
# APPLICATION
# ============================================================================

METHOD_LABELS = {
    'mean': 'Mean',
    'median': 'Median',
    'most_frequent': 'Most Frequent (Mode)',
    'constant': 'Constant Value',
    'knn': 'KNN Imputer',
    'iterative': 'Iterative Imputer',
}


def apply_imputation(
    df: pd.DataFrame,
    column_config: Optional[Dict[str, str]] = None,
    default_numeric: str = 'mean',
    default_categorical: str = 'most_frequent',
    constant_values: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Apply imputation to a DataFrame using per-column configuration.

    This function FITS imputers on the given data (training mode).
    For prediction mode, use PreprocessingPipeline.transform() instead.

    Args:
        df: Input DataFrame
        column_config: Dict mapping column name → method string.
                       e.g. {'age': 'median', 'city': 'most_frequent'}
                       If None, auto-recommends per column.
        default_numeric: Fallback method for numeric columns not in column_config
        default_categorical: Fallback method for categorical columns not in column_config
        constant_values: Dict mapping column name → fill value for 'constant' method

    Returns:
        DataFrame with missing values imputed. Original DataFrame is NOT modified.
    """
    df_out = df.copy()
    column_config = column_config or {}
    constant_values = constant_values or {}

    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue

        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        method = column_config.get(col)

        if not method:
            # Auto-recommend if not specified
            analysis = analyze_column_for_imputation(df[col])
            method = analysis['recommended']

        try:
            if method == 'mean':
                if is_numeric:
                    df_out[col] = df[col].fillna(df[col].mean())
                else:
                    mode_val = df[col].mode()
                    df_out[col] = df[col].fillna(mode_val.iloc[0] if not mode_val.empty else 'Unknown')

            elif method == 'median':
                if is_numeric:
                    df_out[col] = df[col].fillna(df[col].median())
                else:
                    mode_val = df[col].mode()
                    df_out[col] = df[col].fillna(mode_val.iloc[0] if not mode_val.empty else 'Unknown')

            elif method == 'most_frequent':
                mode_val = df[col].mode()
                df_out[col] = df[col].fillna(mode_val.iloc[0] if not mode_val.empty else (0 if is_numeric else 'Unknown'))

            elif method == 'constant':
                fill_val = constant_values.get(col, 0 if is_numeric else 'Unknown')
                df_out[col] = df[col].fillna(fill_val)

            elif method == 'knn':
                if is_numeric:
                    imp = KNNImputer(n_neighbors=5)
                    df_out[[col]] = imp.fit_transform(df[[col]])
                else:
                    mode_val = df[col].mode()
                    df_out[col] = df[col].fillna(mode_val.iloc[0] if not mode_val.empty else 'Unknown')

            elif method == 'iterative':
                if is_numeric:
                    imp = IterativeImputer(max_iter=10, random_state=42)
                    df_out[[col]] = imp.fit_transform(df[[col]])
                else:
                    mode_val = df[col].mode()
                    df_out[col] = df[col].fillna(mode_val.iloc[0] if not mode_val.empty else 'Unknown')

            else:
                # Unknown method fallback
                if is_numeric:
                    df_out[col] = df[col].fillna(df[col].mean())
                else:
                    mode_val = df[col].mode()
                    df_out[col] = df[col].fillna(mode_val.iloc[0] if not mode_val.empty else 'Unknown')

        except Exception as e:
            logger.warning(f"Imputation failed for column '{col}' with method '{method}': {e}. Using fallback.")
            if is_numeric:
                df_out[col] = df[col].fillna(df[col].mean())
            else:
                df_out[col] = df[col].fillna('Unknown')

    return df_out


def imputation_config_from_post(post_data: dict) -> Dict[str, str]:
    """
    Parse imputation configuration from a Django POST request dict.

    Expected POST fields:
        impute_{col_name} = method_string

    Args:
        post_data: request.POST dict

    Returns:
        Dict mapping column name → method string
    """
    config = {}
    for key, value in post_data.items():
        if key.startswith('impute_') and value:
            col_name = key[len('impute_'):]
            config[col_name] = value
    return config
