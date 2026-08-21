"""
preprocessing/encoding.py
==========================
Advanced Encoding Analysis and Application for InsightForge.

This module provides:
    1. Per-feature encoding recommendation based on cardinality & target type
    2. Configurable encoding application (accepts user overrides)
    3. A summary table for the UI (feature name, dtype, unique count, recommendation, reason)

ENCODING RECOMMENDATION LOGIC:
    Target column       → NEVER encoded (always preserved)
    Binary (2 unique)   → Label Encoding (simple 0/1)
    Low cardinality
      (3–10 unique)     → One Hot Encoding (creates indicator columns)
    Medium cardinality
      (11–50 unique)    → Ordinal Encoding (preserves cardinality without explosion)
    High cardinality
      (> 50 unique)     → Frequency Encoding (stable dimensionality)
    Ordinal intent      → Ordinal Encoding (e.g., 'low'/'medium'/'high')
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder

logger = logging.getLogger(__name__)


# ============================================================================
# ENCODING LABELS (for UI display)
# ============================================================================

ENCODING_LABELS = {
    'onehot':    'One Hot Encoding',
    'ordinal':   'Ordinal Encoding',
    'label':     'Label Encoding',
    'frequency': 'Frequency Encoding',
    'binary':    'Binary Encoding',
    'none':      'No Encoding (already numeric)',
}

ORDINAL_HINT_WORDS = {
    'low', 'medium', 'high', 'very low', 'very high',
    'small', 'large', 'poor', 'fair', 'good', 'excellent',
    'never', 'rarely', 'sometimes', 'often', 'always',
    'none', 'some', 'most', 'all',
    'grade', 'level', 'rank', 'tier', 'class',
}


# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_column_for_encoding(
    series: pd.Series,
    target_col: str,
    target_dtype: Optional[str] = None,
) -> Dict:
    """
    Analyze a single column and return encoding recommendation.

    Args:
        series: A pandas Series (single column)
        target_col: Name of the target column (must never be encoded)
        target_dtype: dtype string of the target column (for context)

    Returns:
        dict with keys:
            column          (str)   : Column name
            dtype           (str)   : Data type
            n_unique        (int)   : Number of unique values
            is_target       (bool)  : Whether this is the target column
            needs_encoding  (bool)  : Whether encoding is needed
            recommended     (str)   : Recommended encoding method key
            recommended_label (str) : Human-readable label
            reason          (str)   : Explanation
            sample_values   (list)  : Up to 5 sample values
    """
    col_name = series.name
    is_target = (col_name == target_col)
    is_numeric = pd.api.types.is_numeric_dtype(series)
    n_unique = series.nunique()
    dtype_str = str(series.dtype)
    sample_values = series.dropna().unique()[:5].tolist()

    # Target column — never encode
    if is_target:
        return {
            'column': col_name,
            'dtype': dtype_str,
            'n_unique': n_unique,
            'is_target': True,
            'needs_encoding': False,
            'recommended': 'none',
            'recommended_label': '🔒 Target Column — Not Encoded',
            'reason': 'This is the target (label) column. It is never encoded.',
            'sample_values': sample_values,
        }

    # Already numeric — no encoding needed
    if is_numeric:
        return {
            'column': col_name,
            'dtype': dtype_str,
            'n_unique': n_unique,
            'is_target': False,
            'needs_encoding': False,
            'recommended': 'none',
            'recommended_label': 'No Encoding (already numeric)',
            'reason': 'Column is already numeric. No encoding required.',
            'sample_values': sample_values,
        }

    # Categorical — determine best encoding
    recommended, reason = _recommend_categorical_encoding(series, n_unique)

    return {
        'column': col_name,
        'dtype': dtype_str,
        'n_unique': n_unique,
        'is_target': False,
        'needs_encoding': True,
        'recommended': recommended,
        'recommended_label': ENCODING_LABELS.get(recommended, recommended),
        'reason': reason,
        'sample_values': [str(v) for v in sample_values],
    }


def analyze_dataframe_for_encoding(df: pd.DataFrame, target_col: str) -> List[Dict]:
    """
    Analyze all columns and return encoding recommendations.

    Args:
        df: Input DataFrame
        target_col: Name of the target column

    Returns:
        List of analysis dicts, with categorical columns first
    """
    results = []
    target_dtype = str(df[target_col].dtype) if target_col in df.columns else None

    for col in df.columns:
        analysis = analyze_column_for_encoding(df[col], target_col, target_dtype)
        results.append(analysis)

    # Sort: target first, then columns needing encoding, then already-numeric
    results.sort(key=lambda x: (not x['is_target'], not x['needs_encoding']))
    return results


def _recommend_categorical_encoding(series: pd.Series, n_unique: int) -> tuple:
    """
    Core recommendation logic for categorical columns.

    Returns:
        (method_key, reason_string)
    """
    # Binary: exactly 2 unique values
    if n_unique == 2:
        return 'label', (
            f"Binary column with exactly 2 categories — "
            "Label Encoding (0/1) is most efficient and interpretable."
        )

    # Check for ordinal-hint words in values
    sample_lower = {str(v).lower().strip() for v in series.dropna().unique()}
    if sample_lower & ORDINAL_HINT_WORDS:
        return 'ordinal', (
            f"Column appears to have ordinal values (e.g., {', '.join(list(sample_lower)[:3])}) — "
            "Ordinal Encoding preserves the natural ordering between categories."
        )

    # Low cardinality: 3–10 unique
    if n_unique <= 10:
        return 'onehot', (
            f"Low cardinality ({n_unique} unique values) — "
            "One Hot Encoding creates indicator columns, which works best with tree and linear models."
        )

    # Medium cardinality: 11–50 unique
    if n_unique <= 50:
        return 'ordinal', (
            f"Medium cardinality ({n_unique} unique values) — "
            "Ordinal Encoding assigns integer codes. More compact than OHE and avoids the curse of dimensionality."
        )

    # High cardinality: > 50 unique
    return 'frequency', (
        f"High cardinality ({n_unique} unique values) — "
        "Frequency Encoding replaces each category with its occurrence frequency. "
        "Avoids dimensionality explosion while capturing category prevalence."
    )


# ============================================================================
# APPLICATION
# ============================================================================

class EncodingApplier:
    """
    Fits and applies encoding transformations to a DataFrame.

    Designed to be called during training:
        1. Call fit_transform(df, target_col, column_config)
        2. Save the EncodingApplier (or the full PreprocessingPipeline) for prediction
        3. Call transform(new_df) during prediction — no re-fitting

    The target column is always preserved unchanged.
    """

    def __init__(self):
        self._encoders: Dict[str, Any] = {}
        self._ohe_feature_names: Dict[str, List[str]] = {}
        self._is_fitted: bool = False
        self.target_col_: str = ''

    def fit_transform(
        self,
        df: pd.DataFrame,
        target_col: str,
        column_config: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """
        Fit encoders and transform the DataFrame.

        Args:
            df: Input DataFrame (features + target)
            target_col: Target column — never encoded
            column_config: Dict mapping col_name → encoding method.
                           If None, auto-recommends per column.

        Returns:
            Encoded DataFrame with target preserved unchanged
        """
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

        self.target_col_ = target_col
        column_config = column_config or {}

        # Separate target
        y = df[target_col].copy()
        X = df.drop(columns=[target_col]).copy()

        # Identify categorical columns
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

        X_out = X.copy()
        ohe_outputs = []
        cols_to_drop = []

        for col in cat_cols:
            n_unique = X[col].nunique()
            analysis = analyze_column_for_encoding(X[col], target_col)
            method = column_config.get(col, analysis['recommended'])

            if method == 'none':
                continue

            X_out, enc_info, ohe_df = self._fit_encode_column(X, X_out, col, method)
            self._encoders[col] = enc_info

            if ohe_df is not None:
                ohe_outputs.append(ohe_df)
                self._ohe_feature_names[col] = list(ohe_df.columns)
                cols_to_drop.append(col)
            elif method in ('binary',):
                cols_to_drop.append(col)

        # Drop original columns that were replaced (OHE, binary)
        X_out = X_out.drop(columns=[c for c in cols_to_drop if c in X_out.columns], errors='ignore')
        if ohe_outputs:
            X_out = pd.concat([X_out] + ohe_outputs, axis=1)

        self._is_fitted = True

        # Reattach target
        result = X_out.copy()
        result[target_col] = y.values
        return result

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted encoders to new data (prediction mode — no re-fitting).
        """
        if not self._is_fitted:
            raise RuntimeError("EncodingApplier must be fitted before calling transform().")

        if self.target_col_ in df.columns:
            y = df[self.target_col_].copy()
            X = df.drop(columns=[self.target_col_]).copy()
        else:
            y = None
            X = df.copy()

        X_out = X.copy()
        ohe_outputs = []
        cols_to_drop = []

        for col, enc_info in self._encoders.items():
            if col not in X.columns:
                continue

            method = enc_info.get('method', 'ordinal')
            enc = enc_info.get('encoder')

            try:
                if method == 'ordinal' or method == 'label':
                    X_out[col] = enc.transform(X[[col]].astype(str)).astype(int)

                elif method == 'onehot':
                    encoded = enc.transform(X[[col]].astype(str))
                    feat_names = self._ohe_feature_names.get(col, [])
                    ohe_df = pd.DataFrame(encoded, columns=feat_names, index=X.index)
                    ohe_outputs.append(ohe_df)
                    cols_to_drop.append(col)

                elif method == 'frequency':
                    freq_map = enc_info.get('freq_map', {})
                    X_out[col] = X[col].map(freq_map).fillna(0.0)

                elif method == 'binary':
                    # Re-apply binary encoding from stored mapping
                    code_map = enc_info.get('code_map', {})
                    n_bits = enc_info.get('n_bits', 1)
                    bin_col_names = enc_info.get('bin_cols', [])
                    codes = X[col].map(code_map).fillna(-1).astype(int)
                    for bit, bcol in enumerate(bin_col_names):
                        X_out[bcol] = ((codes >> bit) & 1).values
                    cols_to_drop.append(col)

            except Exception as e:
                logger.warning(f"Transform encoding failed for '{col}': {e}")

        X_out = X_out.drop(columns=[c for c in cols_to_drop if c in X_out.columns], errors='ignore')
        if ohe_outputs:
            X_out = pd.concat([X_out] + ohe_outputs, axis=1)

        if y is not None:
            X_out[self.target_col_] = y.values
        return X_out

    def _fit_encode_column(self, X_orig, X_out, col, method):
        """Fit and apply a single encoding method. Returns (X_out, enc_info, ohe_df_or_None)."""
        n_unique = X_orig[col].nunique()

        try:
            if method == 'onehot':
                enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
                encoded = enc.fit_transform(X_orig[[col]].astype(str))
                feat_names = [f"{col}_{cat}" for cat in enc.categories_[0][1:]]
                ohe_df = pd.DataFrame(encoded, columns=feat_names, index=X_orig.index)
                enc_info = {'method': 'onehot', 'encoder': enc}
                return X_out, enc_info, ohe_df

            elif method == 'frequency':
                counts = X_orig[col].value_counts(normalize=True)
                freq_map = counts.to_dict()
                X_out[col] = X_orig[col].map(freq_map).fillna(0.0)
                enc_info = {'method': 'frequency', 'freq_map': freq_map, 'encoder': None}
                return X_out, enc_info, None

            elif method == 'binary':
                cats = list(X_orig[col].dropna().unique())
                code_map = {cat: i for i, cat in enumerate(cats)}
                n_bits = max(1, int(np.ceil(np.log2(len(cats) + 1))))
                bin_col_names = [f"{col}_bit{b}" for b in range(n_bits)]
                codes = X_orig[col].map(code_map).fillna(-1).astype(int)
                for bit, bcol in enumerate(bin_col_names):
                    X_out[bcol] = ((codes >> bit) & 1).values
                enc_info = {
                    'method': 'binary',
                    'encoder': None,
                    'code_map': code_map,
                    'n_bits': n_bits,
                    'bin_cols': bin_col_names,
                }
                return X_out, enc_info, None

            else:  # ordinal or label — same implementation
                enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                X_out[col] = enc.fit_transform(X_orig[[col]].astype(str)).astype(int)
                enc_info = {'method': method, 'encoder': enc}
                return X_out, enc_info, None

        except Exception as e:
            logger.warning(f"Encoding column '{col}' with '{method}' failed: {e}. Using ordinal fallback.")
            enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            X_out[col] = enc.fit_transform(X_orig[[col]].astype(str)).astype(int)
            return X_out, {'method': 'ordinal', 'encoder': enc}, None


# ============================================================================
# HELPERS
# ============================================================================

def encoding_config_from_post(post_data: dict, df_columns: list) -> Dict[str, str]:
    """
    Parse encoding configuration from a Django POST request dict.

    Expected POST fields:
        encode_{col_name} = method_string

    Args:
        post_data: request.POST dict
        df_columns: List of column names to validate against

    Returns:
        Dict mapping column name → encoding method string
    """
    config = {}
    for key, value in post_data.items():
        if key.startswith('encode_') and value:
            col_name = key[len('encode_'):]
            if col_name in df_columns:
                config[col_name] = value
    return config
