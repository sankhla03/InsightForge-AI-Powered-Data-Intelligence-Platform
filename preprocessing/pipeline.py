"""
preprocessing/pipeline.py
=========================
Reusable Preprocessing Pipeline for InsightForge.

This is the CORE of Feature 1 — a single serializable object that encapsulates
the entire preprocessing workflow:
    Missing Value Imputation → Encoding → Feature Selection → Feature Scaling

CRITICAL DESIGN RULES:
    1. fit_transform() is called ONCE during training — learns all statistics
    2. transform() is called during prediction — applies same statistics, NO re-fitting
    3. The pipeline is saved alongside the trained model (prevents data leakage)
    4. The target column is ALWAYS excluded from all transformations
    5. Supports joblib serialization for persistence

Usage (Training):
    pipeline = PreprocessingPipeline(config)
    X_processed = pipeline.fit_transform(df, target_col='price')
    pipeline.save('saved_models/pipeline_abc123.pkl')

Usage (Prediction):
    pipeline = PreprocessingPipeline.load('saved_models/pipeline_abc123.pkl')
    X_processed = pipeline.transform(new_input_df)   # No re-fitting!
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401 — required to unlock IterativeImputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    OrdinalEncoder, OneHotEncoder, LabelEncoder,
)
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

IMPUTATION_METHODS = ('mean', 'median', 'most_frequent', 'constant', 'knn', 'iterative')
ENCODING_METHODS = ('ordinal', 'onehot', 'label', 'frequency', 'binary')
SCALING_METHODS = ('standard', 'minmax', 'robust', 'none')
SELECTION_METHODS = ('correlation', 'kbest', 'rfe', 'tree', 'none')


# ============================================================================
# HELPER — Frequency Encoding (not in sklearn)
# ============================================================================

class FrequencyEncoder:
    """Encodes categories by their frequency ratio in the training data."""

    def __init__(self):
        self.freq_maps_: Dict[str, Dict[Any, float]] = {}

    def fit(self, X: pd.DataFrame) -> 'FrequencyEncoder':
        self.freq_maps_ = {}
        for col in X.columns:
            counts = X[col].value_counts(normalize=True)
            self.freq_maps_[col] = counts.to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        for col in X.columns:
            if col in self.freq_maps_:
                X_out[col] = X[col].map(self.freq_maps_[col]).fillna(0.0)
        return X_out

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)


class BinaryEncoder:
    """
    Binary encoding: converts categories to integer codes, then to binary bits.
    More compact than OHE for high-cardinality features.
    """

    def __init__(self):
        self.encoders_: Dict[str, Dict[Any, int]] = {}
        self.n_bits_: Dict[str, int] = {}

    def fit(self, X: pd.DataFrame) -> 'BinaryEncoder':
        self.encoders_ = {}
        self.n_bits_ = {}
        for col in X.columns:
            cats = list(X[col].dropna().unique())
            self.encoders_[col] = {cat: i for i, cat in enumerate(cats)}
            self.n_bits_[col] = max(1, int(np.ceil(np.log2(len(cats) + 1))))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        dfs = []
        for col in X.columns:
            enc = self.encoders_.get(col, {})
            n = self.n_bits_.get(col, 1)
            codes = X[col].map(enc).fillna(-1).astype(int)
            for bit in range(n):
                dfs.append(pd.Series(
                    ((codes >> bit) & 1).values,
                    name=f"{col}_bit{bit}",
                    index=X.index,
                ))
        return pd.concat(dfs, axis=1) if dfs else pd.DataFrame(index=X.index)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)


# ============================================================================
# MAIN PIPELINE CLASS
# ============================================================================

class PreprocessingPipeline:
    """
    A single serializable object that encapsulates the complete preprocessing
    workflow for an ML experiment.

    Attributes:
        config (dict): User-provided or auto-recommended configuration
        feature_columns_ (list): Features after selection (set after fit)
        target_col_ (str): Target column name (set after fit)
        task_type_ (str): 'classification' or 'regression' (set after fit)
        is_fitted_ (bool): True after fit_transform() has been called
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the pipeline with a configuration dictionary.

        Args:
            config: Optional configuration dict. Keys:
                imputation   (dict): {col_name: {'method': 'mean', 'fill_value': None}}
                encoding     (dict): {col_name: {'method': 'ordinal'}}
                scaling      (str):  'standard' | 'minmax' | 'robust' | 'none'
                selection    (dict): {'method': 'kbest', 'k': 10}
        """
        self.config: Dict = config or {}

        # Internal state — populated during fit_transform()
        self.feature_columns_: List[str] = []
        self.target_col_: str = ''
        self.task_type_: str = 'classification'
        self.is_fitted_: bool = False

        # Fitted transformers (stored for transform())
        self._imputers: Dict[str, Any] = {}
        self._encoders: Dict[str, Any] = {}
        self._scaler: Optional[Any] = None
        self._selector: Optional[Any] = None
        self._selector_cols: List[str] = []   # columns fed into selector
        self._ohe_cols: List[str] = []         # columns that became OHE
        self._ohe_feature_names: List[str] = []  # OHE output column names

    # -------------------------------------------------------------------------
    # PUBLIC API
    # -------------------------------------------------------------------------

    def fit_transform(
        self,
        df: pd.DataFrame,
        target_col: str,
        task_type: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fit the pipeline on training data and return transformed DataFrame.

        Args:
            df: Raw input DataFrame (features + target)
            target_col: Name of the target column (never transformed)
            task_type: 'classification' or 'regression'. Auto-detected if None.

        Returns:
            Transformed DataFrame with selected, encoded, scaled features + target
        """
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

        self.target_col_ = target_col
        y = df[target_col].copy()
        X = df.drop(columns=[target_col]).copy()

        # Auto-detect task type
        if task_type:
            self.task_type_ = task_type
        else:
            self.task_type_ = self._detect_task_type(y)

        logger.info("=" * 60)
        logger.info("PreprocessingPipeline.fit_transform()")
        logger.info(f"  Target: {target_col} | Task: {self.task_type_}")
        logger.info(f"  Input shape: {X.shape}")

        # Step 1: Imputation
        X = self._fit_transform_imputation(X)
        logger.info(f"  After imputation: {X.shape}")

        # Step 2: Encoding
        X = self._fit_transform_encoding(X)
        logger.info(f"  After encoding: {X.shape}")

        # Step 3: Feature Selection
        X = self._fit_transform_selection(X, y)
        logger.info(f"  After selection: {X.shape}")

        # Step 4: Scaling
        X = self._fit_transform_scaling(X)
        logger.info(f"  After scaling: {X.shape}")

        self.feature_columns_ = list(X.columns)
        self.is_fitted_ = True

        # Recombine features + target
        result = X.copy()
        result[target_col] = y.values
        logger.info("  Pipeline fit_transform complete.")
        return result

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the fitted pipeline to new data (prediction mode).

        CRITICAL: This never re-fits any transformer. It uses the exact
        statistics learned during fit_transform().

        Args:
            df: Input DataFrame. Must contain the same feature columns.

        Returns:
            Transformed feature DataFrame (no target column)
        """
        if not self.is_fitted_:
            raise RuntimeError(
                "Pipeline is not fitted. Call fit_transform() first."
            )

        # Drop target if present (prediction data may not have it)
        if self.target_col_ in df.columns:
            X = df.drop(columns=[self.target_col_]).copy()
        else:
            X = df.copy()

        # Step 1: Imputation (transform only)
        X = self._transform_imputation(X)

        # Step 2: Encoding (transform only)
        X = self._transform_encoding(X)

        # Step 3: Feature Selection (select known columns)
        X = self._transform_selection(X)

        # Step 4: Scaling (transform only)
        X = self._transform_scaling(X)

        return X

    def save(self, path: str) -> str:
        """Serialize and save the fitted pipeline to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Pipeline saved to: {path}")
        return path

    @staticmethod
    def load(path: str) -> 'PreprocessingPipeline':
        """Load a fitted pipeline from disk."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Pipeline file not found: {path}")
        pipeline = joblib.load(path)
        if not isinstance(pipeline, PreprocessingPipeline):
            raise TypeError("Loaded object is not a PreprocessingPipeline.")
        return pipeline

    def get_config_summary(self) -> Dict:
        """Return a human-readable summary of what the pipeline does."""
        return {
            'target': self.target_col_,
            'task_type': self.task_type_,
            'is_fitted': self.is_fitted_,
            'feature_columns': self.feature_columns_,
            'n_features_selected': len(self.feature_columns_),
            'imputation_config': self.config.get('imputation', {}),
            'encoding_config': self.config.get('encoding', {}),
            'scaling': self.config.get('scaling', 'standard'),
            'selection': self.config.get('selection', {}).get('method', 'none'),
        }

    # -------------------------------------------------------------------------
    # PRIVATE: FIT + TRANSFORM
    # -------------------------------------------------------------------------

    def _fit_transform_imputation(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit imputers on each column and transform."""
        self._imputers = {}
        imputation_config = self.config.get('imputation', {})
        X_out = X.copy()

        for col in X.columns:
            if X[col].isnull().sum() == 0:
                continue  # No missing values — skip

            col_config = imputation_config.get(col, {})
            # Support both str shorthand ('median') and dict ({'method': 'median'})
            if isinstance(col_config, str):
                method = col_config
                fill_value = 0
            else:
                method = col_config.get('method', self._recommend_imputation_method(X[col]))
                fill_value = col_config.get('fill_value', 0)

            try:
                if method == 'knn':
                    imp = KNNImputer(n_neighbors=5)
                    # KNNImputer needs numeric input
                    if pd.api.types.is_numeric_dtype(X[col]):
                        X_out[[col]] = imp.fit_transform(X[[col]])
                        self._imputers[col] = imp
                    else:
                        # Fallback for non-numeric
                        mode_val = X[col].mode()
                        fill = mode_val.iloc[0] if not mode_val.empty else 'Unknown'
                        X_out[col] = X[col].fillna(fill)
                        self._imputers[col] = ('constant', fill)

                elif method == 'iterative':
                    if pd.api.types.is_numeric_dtype(X[col]):
                        imp = IterativeImputer(max_iter=10, random_state=42)
                        X_out[[col]] = imp.fit_transform(X[[col]])
                        self._imputers[col] = imp
                    else:
                        mode_val = X[col].mode()
                        fill = mode_val.iloc[0] if not mode_val.empty else 'Unknown'
                        X_out[col] = X[col].fillna(fill)
                        self._imputers[col] = ('constant', fill)

                elif method == 'constant':
                    X_out[col] = X[col].fillna(fill_value)
                    self._imputers[col] = ('constant', fill_value)

                elif method in ('mean', 'median', 'most_frequent'):
                    strategy = method
                    imp = SimpleImputer(strategy=strategy)
                    X_out[[col]] = imp.fit_transform(X[[col]].astype(str if method == 'most_frequent' else float))
                    self._imputers[col] = imp

                else:
                    # Default: use mean for numeric, most_frequent for categorical
                    if pd.api.types.is_numeric_dtype(X[col]):
                        X_out[col] = X[col].fillna(X[col].mean())
                        self._imputers[col] = ('constant', X[col].mean())
                    else:
                        mode_val = X[col].mode()
                        fill = mode_val.iloc[0] if not mode_val.empty else 'Unknown'
                        X_out[col] = X[col].fillna(fill)
                        self._imputers[col] = ('constant', fill)

            except Exception as e:
                logger.warning(f"Imputation failed for '{col}' with method '{method}': {e}. Using fallback.")
                if pd.api.types.is_numeric_dtype(X[col]):
                    X_out[col] = X[col].fillna(X[col].mean())
                else:
                    X_out[col] = X[col].fillna('Unknown')

        return X_out

    def _transform_imputation(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted imputers to new data."""
        X_out = X.copy()
        for col, imp in self._imputers.items():
            if col not in X.columns:
                continue
            if X[col].isnull().sum() == 0:
                continue
            try:
                if isinstance(imp, tuple) and imp[0] == 'constant':
                    X_out[col] = X[col].fillna(imp[1])
                elif isinstance(imp, (SimpleImputer, KNNImputer, IterativeImputer)):
                    X_out[[col]] = imp.transform(X[[col]])
            except Exception as e:
                logger.warning(f"Transform imputation failed for '{col}': {e}")
        return X_out

    def _fit_transform_encoding(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit encoders on categorical columns and transform."""
        self._encoders = {}
        self._ohe_cols = []
        self._ohe_feature_names = []
        encoding_config = self.config.get('encoding', {})
        X_out = X.copy()

        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

        ohe_outputs = []
        cols_to_drop = []

        for col in cat_cols:
            col_config = encoding_config.get(col, {})
            n_unique = X[col].nunique()
            # Support both str shorthand ('onehot') and dict ({'method': 'onehot'})
            if isinstance(col_config, str):
                method = col_config
            else:
                method = col_config.get('method', self._recommend_encoding_method(X[col], n_unique))

            try:
                if method == 'onehot':
                    enc = OneHotEncoder(
                        sparse_output=False,
                        handle_unknown='ignore',
                        drop='first',
                    )
                    encoded = enc.fit_transform(X[[col]].astype(str))
                    feat_names = [f"{col}_{cat}" for cat in enc.categories_[0][1:]]
                    ohe_df = pd.DataFrame(encoded, columns=feat_names, index=X.index)
                    ohe_outputs.append(ohe_df)
                    self._ohe_cols.append(col)
                    self._ohe_feature_names.extend(feat_names)
                    self._encoders[col] = ('onehot', enc)
                    cols_to_drop.append(col)

                elif method == 'frequency':
                    enc = FrequencyEncoder()
                    X_out[col] = enc.fit(X[[col]]).transform(X[[col]])[col]
                    self._encoders[col] = ('frequency', enc)

                elif method == 'binary':
                    enc = BinaryEncoder()
                    bin_df = enc.fit_transform(X[[col]])
                    # Add binary columns
                    for bcol in bin_df.columns:
                        X_out[bcol] = bin_df[bcol].values
                    self._encoders[col] = ('binary', enc, list(bin_df.columns))
                    cols_to_drop.append(col)

                elif method == 'label':
                    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                    X_out[col] = enc.fit_transform(X[[col]].astype(str)).astype(int)
                    self._encoders[col] = ('ordinal', enc)

                else:  # ordinal (default)
                    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                    X_out[col] = enc.fit_transform(X[[col]].astype(str)).astype(int)
                    self._encoders[col] = ('ordinal', enc)

            except Exception as e:
                logger.warning(f"Encoding failed for '{col}' method '{method}': {e}. Using ordinal fallback.")
                enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                try:
                    X_out[col] = enc.fit_transform(X[[col]].astype(str)).astype(int)
                    self._encoders[col] = ('ordinal', enc)
                except Exception:
                    X_out[col] = 0  # Last resort: zero-fill
                    self._encoders[col] = ('constant', 0)

        # Drop original OHE/binary columns and add encoded versions
        X_out = X_out.drop(columns=[c for c in cols_to_drop if c in X_out.columns], errors='ignore')
        if ohe_outputs:
            ohe_combined = pd.concat(ohe_outputs, axis=1)
            X_out = pd.concat([X_out, ohe_combined], axis=1)

        return X_out

    def _transform_encoding(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted encoders to new data."""
        X_out = X.copy()
        cols_to_drop = []
        ohe_outputs = []

        for col, enc_info in self._encoders.items():
            if col not in X.columns:
                continue
            enc_type = enc_info[0]
            try:
                if enc_type == 'ordinal':
                    enc = enc_info[1]
                    X_out[col] = enc.transform(X[[col]].astype(str)).astype(int)

                elif enc_type == 'onehot':
                    enc = enc_info[1]
                    encoded = enc.transform(X[[col]].astype(str))
                    feat_names = [f"{col}_{cat}" for cat in enc.categories_[0][1:]]
                    ohe_df = pd.DataFrame(encoded, columns=feat_names, index=X.index)
                    ohe_outputs.append(ohe_df)
                    cols_to_drop.append(col)

                elif enc_type == 'frequency':
                    enc = enc_info[1]
                    X_out[col] = enc.transform(X[[col]])[col]

                elif enc_type == 'binary':
                    enc = enc_info[1]
                    bin_df = enc.transform(X[[col]])
                    for bcol in bin_df.columns:
                        X_out[bcol] = bin_df[bcol].values if bcol in bin_df.columns else 0
                    cols_to_drop.append(col)

                elif enc_type == 'constant':
                    X_out[col] = enc_info[1]

            except Exception as e:
                logger.warning(f"Transform encoding failed for '{col}': {e}")

        X_out = X_out.drop(columns=[c for c in cols_to_drop if c in X_out.columns], errors='ignore')
        if ohe_outputs:
            ohe_combined = pd.concat(ohe_outputs, axis=1)
            X_out = pd.concat([X_out, ohe_combined], axis=1)

        return X_out

    def _fit_transform_selection(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit feature selector on training data and return selected features."""
        selection_config = self.config.get('selection', {})
        method = selection_config.get('method', 'none')
        k = min(selection_config.get('k', 10), len(X.columns))

        self._selector = None
        self._selector_cols = list(X.columns)

        if method == 'none' or k <= 0 or len(X.columns) <= k:
            return X

        # Ensure X is numeric before selection
        X_num = X.select_dtypes(include=[np.number])
        if X_num.empty or len(X_num.columns) == 0:
            return X

        try:
            if method == 'kbest':
                score_func = f_classif if self.task_type_ == 'classification' else f_regression
                sel = SelectKBest(score_func=score_func, k=min(k, len(X_num.columns)))
                sel.fit(X_num, y)
                selected_cols = X_num.columns[sel.get_support()].tolist()
                self._selector = sel
                self._selector_cols = X_num.columns.tolist()
                return X[selected_cols]

            elif method == 'correlation':
                corr = X_num.corrwith(y).abs().sort_values(ascending=False)
                selected_cols = corr.head(k).index.tolist()
                self._selector = ('correlation', selected_cols)
                self._selector_cols = selected_cols
                return X[selected_cols]

            elif method == 'rfe':
                estimator = (
                    LogisticRegression(max_iter=500, random_state=42)
                    if self.task_type_ == 'classification'
                    else RandomForestRegressor(n_estimators=50, random_state=42)
                )
                sel = RFE(estimator=estimator, n_features_to_select=min(k, len(X_num.columns)))
                sel.fit(X_num, y)
                selected_cols = X_num.columns[sel.support_].tolist()
                self._selector = sel
                self._selector_cols = X_num.columns.tolist()
                return X[selected_cols]

            elif method == 'tree':
                estimator = (
                    RandomForestClassifier(n_estimators=100, random_state=42)
                    if self.task_type_ == 'classification'
                    else RandomForestRegressor(n_estimators=100, random_state=42)
                )
                estimator.fit(X_num, y)
                importances = pd.Series(estimator.feature_importances_, index=X_num.columns)
                selected_cols = importances.nlargest(k).index.tolist()
                self._selector = ('tree', selected_cols)
                self._selector_cols = selected_cols
                return X[selected_cols]

        except Exception as e:
            logger.warning(f"Feature selection failed with method '{method}': {e}. Using all features.")
            return X

        return X

    def _transform_selection(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply feature selection to new data using stored column list."""
        if self._selector is None:
            return X

        if isinstance(self._selector, tuple):
            # Correlation or tree — just filter columns
            selected_cols = self._selector[1]
        elif hasattr(self._selector, 'get_support'):
            # SelectKBest or RFE
            X_num = X[self._selector_cols] if all(c in X.columns for c in self._selector_cols) else X
            selected_cols = [
                col for col, sup in zip(self._selector_cols, self._selector.get_support())
                if sup
            ]
        else:
            return X

        # Return only columns that exist in X
        available = [c for c in selected_cols if c in X.columns]
        return X[available] if available else X

    def _fit_transform_scaling(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit scaler on training data and return scaled features."""
        method = self.config.get('scaling', 'standard')
        if method == 'none':
            self._scaler = None
            return X

        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            self._scaler = None
            return X

        scaler_map = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler(),
        }
        self._scaler = scaler_map.get(method, StandardScaler())

        X_out = X.copy()
        X_out[numeric_cols] = self._scaler.fit_transform(X[numeric_cols])
        return X_out

    def _transform_scaling(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted scaler to new data."""
        if self._scaler is None:
            return X

        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return X

        X_out = X.copy()
        try:
            X_out[numeric_cols] = self._scaler.transform(X[numeric_cols])
        except Exception as e:
            logger.warning(f"Scaling transform failed: {e}")
        return X_out

    # -------------------------------------------------------------------------
    # PRIVATE: RECOMMENDATION HELPERS
    # -------------------------------------------------------------------------

    @staticmethod
    def _detect_task_type(y: pd.Series, threshold: int = 10) -> str:
        """Detect classification vs regression based on target properties."""
        if pd.api.types.is_float_dtype(y):
            return 'regression'
        if pd.api.types.is_numeric_dtype(y) and y.nunique() >= threshold:
            return 'regression'
        return 'classification'

    @staticmethod
    def _recommend_imputation_method(series: pd.Series) -> str:
        """Recommend imputation method based on column characteristics."""
        if not pd.api.types.is_numeric_dtype(series):
            return 'most_frequent'
        skewness = abs(series.skew())
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        n_outliers = ((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum()
        outlier_pct = n_outliers / max(len(series), 1)
        if outlier_pct > 0.05 or skewness > 1.0:
            return 'median'
        return 'mean'

    @staticmethod
    def _recommend_encoding_method(series: pd.Series, n_unique: int) -> str:
        """Recommend encoding based on cardinality."""
        if n_unique == 2:
            return 'label'
        if n_unique <= 10:
            return 'onehot'
        if n_unique <= 50:
            return 'ordinal'
        return 'frequency'


# ============================================================================
# FACTORY: Build pipeline from session config
# ============================================================================

def build_pipeline_from_session(request, target_col: str) -> PreprocessingPipeline:
    """
    Construct a PreprocessingPipeline from the current session state.
    Reads imputation, encoding, scaling, and selection config from session.

    Args:
        request: Django HttpRequest with session data
        target_col: Target column name

    Returns:
        Configured (not yet fitted) PreprocessingPipeline
    """
    config = {
        'imputation': request.session.get('imputation_config', {}),
        'encoding': request.session.get('encoding_config', {}),
        'scaling': request.session.get('scaler_params', {}).get('method', 'standard'),
        'selection': {
            'method': request.session.get('feature_selection_method', 'none'),
            'k': request.session.get('feature_selection_k', 10),
        },
    }
    return PreprocessingPipeline(config)


def get_pipeline_save_path(base_dir: str, session_key: str) -> str:
    """Generate a standardized path for saving the preprocessing pipeline."""
    models_dir = os.path.join(base_dir, 'saved_models')
    os.makedirs(models_dir, exist_ok=True)
    return os.path.join(models_dir, f"pipeline_{session_key or 'default'}.pkl")
