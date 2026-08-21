import pandas as pd
import numpy as np

from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Import centralized encoding function from utils
from .utils import convert_features_to_numeric


# ============================================================================
# CENTRALIZED FEATURE SELECTION FUNCTION
# ============================================================================
def run_feature_selection(df, target_col, method, k=5):
    """
    CENTRALIZED entry point for ALL feature selection operations.
    
    MANDATORY REQUIREMENTS:
    1. target_col MUST be passed explicitly - NO implicit selection
    2. Logs target used: print("FEATURE SELECTION TARGET:", target_col)
    3. HARD FAILS if target_col not in df.columns
    4. NO fallback logic (no columns[-1], no numeric_cols[0])
    5. Target passed explicitly to ALL internal methods
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe (should already be cleaned/encoded)
    target_col : str
        Name of target column - MUST be explicitly passed
    method : str
        Feature selection method: "correlation", "kbest", "rfe", "tree", "none"
    k : int
        Number of features to select (default: 5)
    
    Returns:
    --------
    dict with keys:
        - 'scores': dict of feature -> score
        - 'selected': list of selected feature names
        - 'target': the validated target column name
        - 'method': the method used
    
    Raises:
    -------
    ValueError
        If target_col not in df.columns
        If unknown method specified
    """
    # ========================================================================
    # STEP 1: HARD FAIL if target missing - NO FALLBACK
    # ========================================================================
    if target_col not in df.columns:
        available_cols = list(df.columns)
        error_msg = (
            f"\n{'=' * 70}\n"
            f"CRITICAL ERROR: TARGET COLUMN NOT FOUND\n"
            f"{'=' * 70}\n"
            f"Requested target: '{target_col}'\n"
            f"Available columns: {available_cols}\n"
            f"\nACTION REQUIRED:\n"
            f"  - User must select a valid target column from the dropdown\n"
            f"  - Target column must exist in the dataset\n"
            f"  - Cannot use implicit/default target selection\n"
            f"{'=' * 70}\n"
        )
        print(error_msg)
        raise ValueError(
            f"TARGET COLUMN '{target_col}' NOT FOUND in dataset.\n"
            f"Available columns: {available_cols}"
        )
    
    # ========================================================================
    # STEP 2: MANDATORY LOGGING - Show exact target being used
    # ========================================================================
    print("\n" + "=" * 70)
    print("🔍 FEATURE SELECTION TARGET (EXPLICIT):", target_col)
    print("=" * 70)
    print(f"Method: {method}")
    print(f"K: {k}")
    print(f"Dataset shape: {df.shape}")
    print(f"All columns: {list(df.columns)}")
    print("=" * 70 + "\n")
    
    # ========================================================================
    # STEP 3: Route to appropriate method with EXPLICIT target
    # ========================================================================
    if method == "correlation":
        scores = _correlation_with_target_explicit(df, target_col)
        selected = _get_top_k_from_scores(scores, k)
    elif method == "kbest":
        scores, selected = _select_k_best_explicit(df, target_col, k)
    elif method == "rfe":
        scores, selected = _rfe_selection_explicit(df, target_col, k)
    elif method == "tree":
        scores, selected = _tree_based_explicit(df, target_col, k)
    elif method == "none":
        # Return all features (except target)
        feature_cols = [c for c in df.columns if c != target_col]
        scores = {c: 0.0 for c in feature_cols}
        selected = feature_cols
    else:
        raise ValueError(f"Unknown feature selection method: '{method}'")
    
    # ========================================================================
    # STEP 4: Return structured result
    # ========================================================================
    return {
        'scores': scores,
        'selected': selected,
        'target': target_col,
        'method': method
    }


# ============================================================================
# INTERNAL METHODS - All use EXPLICIT target
# ============================================================================
def _correlation_with_target_explicit(df, target_col):
    """
    Correlation with target - EXPLICIT target version.
    Target must be numeric.
    """
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        raise ValueError(
            f"Correlation method requires numeric target. "
            f"Target '{target_col}' has dtype: {df[target_col].dtype}"
        )
    
    # Use centralized encoding (target excluded)
    X_encoded, y_unchanged, _ = convert_features_to_numeric(df, target_col)
    
    # Get numeric feature columns only
    numeric_cols = X_encoded.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        return {}
    
    # Compute correlation
    corr_df = X_encoded[numeric_cols].copy()
    corr_df[target_col] = y_unchanged.values
    
    corr = corr_df.corr()[target_col]
    corr = corr.drop(target_col).abs().sort_values(ascending=False)
    
    return corr.to_dict()


def _select_k_best_explicit(df, target_col, k):
    """SelectKBest with EXPLICIT target."""
    from sklearn.feature_selection import f_classif, SelectKBest
    
    # Use centralized encoding (target excluded)
    X_encoded, y_unchanged, _ = convert_features_to_numeric(df, target_col)
    
    # Get feature columns (exclude target)
    feature_cols = [c for c in X_encoded.columns if c != target_col]
    X = X_encoded[feature_cols]
    y = y_unchanged
    
    # Handle non-numeric target
    if not pd.api.types.is_numeric_dtype(y):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))
    
    # Select method based on target type
    if y.nunique() <= 10:
        selector = SelectKBest(score_func=f_classif, k=min(k, len(feature_cols)))
    else:
        selector = SelectKBest(score_func=f_regression, k=min(k, len(feature_cols)))
    
    selector.fit(X, y)
    
    # Build scores dict
    scores = {}
    for i, feat in enumerate(feature_cols):
        scores[feat] = float(selector.scores_[i])
    
    # Get selected features
    selected_mask = selector.get_support()
    selected = [f for f, s in zip(feature_cols, selected_mask) if s]
    
    return scores, selected


def _rfe_selection_explicit(df, target_col, k):
    """RFE with EXPLICIT target."""
    from sklearn.feature_selection import RFE
    from sklearn.ensemble import RandomForestClassifier
    
    # Use centralized encoding (target excluded)
    X_encoded, y_unchanged, _ = convert_features_to_numeric(df, target_col)
    
    feature_cols = [c for c in X_encoded.columns if c != target_col]
    X = X_encoded[feature_cols]
    y = y_unchanged
    
    # Handle non-numeric target
    if not pd.api.types.is_numeric_dtype(y):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))
    
    # Use RandomForest for RFE
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rfe = RFE(estimator=rf, n_features_to_select=min(k, len(feature_cols)))
    rfe.fit(X, y)
    
    # Build scores (rank) dict
    scores = {}
    for i, feat in enumerate(feature_cols):
        scores[feat] = float(rfe.ranking_[i])  # Lower rank = better
    
    # Get selected features
    selected = [f for f, s in zip(feature_cols, rfe.support_) if s]
    
    return scores, selected


def _tree_based_explicit(df, target_col, k):
    """Tree-based feature importance with EXPLICIT target."""
    from sklearn.ensemble import RandomForestClassifier
    
    # Use centralized encoding (target excluded)
    X_encoded, y_unchanged, _ = convert_features_to_numeric(df, target_col)
    
    feature_cols = [c for c in X_encoded.columns if c != target_col]
    X = X_encoded[feature_cols]
    y = y_unchanged
    
    # Handle non-numeric target
    if not pd.api.types.is_numeric_dtype(y):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))
    
    # Train RandomForest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Normalize importance scores
    importance_sum = rf.feature_importances_.sum()
    scores = {}
    for i, feat in enumerate(feature_cols):
        normalized = float(rf.feature_importances_[i]) / importance_sum if importance_sum > 0 else 0
        scores[feat] = round(normalized, 4)
    
    # Sort by importance and get top k
    sorted_idx = np.argsort(rf.feature_importances_)[::-1]
    selected = [feature_cols[i] for i in sorted_idx[:k]]
    
    return scores, selected


def _get_top_k_from_scores(scores_dict, k):
    """Helper to get top k features from scores dict."""
    sorted_items = sorted(scores_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    return [feat for feat, score in sorted_items[:k]]


# =====================================================
# 1️⃣ CORRELATION WITH TARGET
# =====================================================
def correlation_with_target(df, target):
    """
    Returns correlation of numeric features with target.
    Target must be numeric.
    
    Uses the centralized convert_features_to_numeric function from utils.py
    to ensure consistent encoding behavior across all feature selection methods.
    """
    # First check if target is numeric
    if not pd.api.types.is_numeric_dtype(df[target]):
        # Return empty series if target is not numeric
        return pd.Series(dtype=float)
    
    # Use centralized encoding function (TARGET EXCLUDED)
    # convert_features_to_numeric returns (X_encoded, y_unchanged, encoding_info)
    # X_encoded is features-only (target already dropped)
    X_encoded, y_unchanged, encoding_info = convert_features_to_numeric(df, target)

    # Get only numeric columns for correlation calculation
    # X_encoded already excludes target, so we use it directly
    numeric_cols = X_encoded.select_dtypes(include=[np.number]).columns.tolist()
    
    # If no numeric feature columns, return empty series
    if not numeric_cols:
        return pd.Series(dtype=float)
    
    # Compute correlation between numeric features and target
    # Create temporary dataframe with encoded features and target
    corr_df = X_encoded[numeric_cols].copy()
    corr_df[target] = y_unchanged.values
    
    corr = corr_df.corr()[target]
    # Remove self-correlation (target with itself) and get absolute values
    corr = corr.drop(target).abs().sort_values(ascending=False)

    return corr


# =====================================================
# 2️⃣ SELECT K BEST
# =====================================================
def select_top_k_features(df, target, k=5):
    """
    Selects top-k features using statistical tests
    (ANOVA for classification, F-regression for regression)
    
    Uses the centralized convert_features_to_numeric function from utils.py
    to ensure consistent encoding behavior across all feature selection methods.
    """
    # Use centralized encoding function (TARGET EXCLUDED)
    # convert_features_to_numeric returns (X_encoded, y_unchanged, encoding_info)
    # X_encoded is features-only (target already dropped)
    X_encoded, y_unchanged, encoding_info = convert_features_to_numeric(df, target)

    # X_encoded already excludes target, so use it directly
    X = X_encoded
    y = y_unchanged

    if y.nunique() <= 10:
        selector = SelectKBest(score_func=f_classif, k=k)
    else:
        selector = SelectKBest(score_func=f_regression, k=k)

    selector.fit(X, y)

    selected_cols = X.columns[selector.get_support()].tolist()
    # Add target column for the final dataframe
    selected_cols.append(target)

    return df[selected_cols]


# =====================================================
# 3️⃣ RFE (Recursive Feature Elimination)
# =====================================================
def rfe_selection(df, target, k=5):
    """
    Selects features using Recursive Feature Elimination
    with Logistic Regression.
    
    Uses the centralized convert_features_to_numeric function from utils.py
    to ensure consistent encoding behavior across all feature selection methods.
    """
    # Use centralized encoding function (TARGET EXCLUDED)
    # convert_features_to_numeric returns (X_encoded, y_unchanged, encoding_info)
    # X_encoded is features-only (target already dropped)
    X_encoded, y_unchanged, encoding_info = convert_features_to_numeric(df, target)

    # X_encoded already excludes target, so use it directly
    X = X_encoded
    y = y_unchanged

    model = LogisticRegression(max_iter=2000)

    rfe = RFE(
        estimator=model,
        n_features_to_select=k
    )
    rfe.fit(X, y)

    selected_cols = X.columns[rfe.support_].tolist()
    # Add target column for the final dataframe
    selected_cols.append(target)

    return df[selected_cols]