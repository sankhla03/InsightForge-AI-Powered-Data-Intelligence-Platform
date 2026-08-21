"""
================================================================================
IMPROVED CORRELATION-BASED FEATURE SELECTION
================================================================================

This module provides enhanced correlation-based feature selection with:
1. Automatic removal of identifier columns (loan_id, *_id, etc.)
2. Absolute correlation calculation
3. Normalized importance scores (0-100%)
4. Proper sorting by normalized importance

Usage:
    from .correlation_fix import compute_correlation_importance
    
    scores, sorted_features, removed_ids = compute_correlation_importance(
        df=df, 
        target_col='loanamount',
        feature_cols=[list of feature columns]
   )
================================================================================
"""

import pandas as pd
import numpy as np
import re


def is_identifier_column(col_name):
    """
    Check if a column name indicates an identifier that should be excluded
    from feature selection.
    
    Identifiers are columns that:
    - Have 'id' in their name (loan_id, customer_id, id, etc.)
    - Have '_no' or 'no_' in their name (loan_no, account_number, etc.)
    - Are specifically known identifier patterns
    
    Args:
        col_name: Column name to check
        
    Returns:
        bool: True if column is an identifier, False otherwise
    """
    col_lower = col_name.lower()
    
    # Identifier patterns
    identifier_patterns = [
        r'.*_?id$',           # loan_id, customer_id, user_id, id
        r'^id_?',             # id, id_number, id_card
        r'.*_no$',            # loan_no, account_no, serial_no
        r'^no_?',             # number, no_number
    ]
    
    for pattern in identifier_patterns:
        if re.search(pattern, col_lower):
            return True
    
    # Specific known identifier columns
    known_identifiers = [
        'loan_id', 'customer_id', 'user_id', 'application_id',
        'id', 'loan_number', 'account_number', 'serial', 
        'index', 'idx', 'record_id', 'case_id'
    ]
    
    if col_lower in known_identifiers:
        return True
    
    return False


def compute_correlation_importance(df, target_col, feature_cols=None):
    """
    Compute normalized correlation-based feature importance scores.
    
    This function:
    1. Removes identifier columns from consideration
    2. Computes Pearson correlation with target
    3. Takes absolute values of correlations
    4. Normalizes to percentage (0-100%) where top feature = 100%
    5. Returns sorted features by normalized importance
    
    Args:
        df: pandas DataFrame with the dataset
        target_col: Name of the target column
        feature_cols: List of feature columns to consider (if None, uses all numeric except target)
        
    Returns:
        dict: normalized_scores - {feature_name: normalized_importance (0-100)}
        list: sorted_features - features sorted by normalized importance (highest first)
        list: removed_identifiers - list of identifier columns that were removed
    """
    
    # Step 1: Identify and remove identifier columns
    removed_identifiers = []
    valid_features = []
    
    if feature_cols is None:
        # Use all numeric columns except target
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c != target_col]
    
    for feat in feature_cols:
        if is_identifier_column(feat):
            removed_identifiers.append(feat)
        else:
            valid_features.append(feat)
    
    # Step 2: Compute correlations
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")
    
    # Check if target is numeric
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        raise ValueError(f"Target column '{target_col}' must be numeric for correlation")
    
    raw_correlations = {}
    for feat in valid_features:
        if feat in df.columns and pd.api.types.is_numeric_dtype(df[feat]):
            corr = df[feat].corr(df[target_col])
            if not pd.isna(corr):
                raw_correlations[feat] = corr
    
    # Step 3: Compute absolute correlations
    abs_correlations = {feat: abs(score) for feat, score in raw_correlations.items()}
    
    # Step 4: Normalize to percentage
    # Formula: normalized = (abs_corr / max_abs_corr) * 100
    max_abs_corr = max(abs_correlations.values()) if abs_correlations else 1.0
    if max_abs_corr == 0:
        max_abs_corr = 1.0  # Avoid division by zero
    
    normalized_scores = {}
    for feat, abs_score in abs_correlations.items():
        normalized_scores[feat] = round((abs_score / max_abs_corr) * 100, 2)
    
    # Step 5: Sort by normalized importance (highest first)
    sorted_features_with_scores = sorted(
        normalized_scores.items(),
        key=lambda x: x[1],  # Higher normalized score = higher importance
        reverse=True
    )
    sorted_features = [f for f, s in sorted_features_with_scores]
    
    return normalized_scores, sorted_features, removed_identifiers


# Example usage and testing
if __name__ == "__main__":
    # Demo with sample data
    print("=" * 70)
    print("IMPROVED CORRELATION-BASED FEATURE SELECTION")
    print("=" * 70)
    
    # Simulated scenario
    sample_features = [
        'loan_id', 'gender', 'married', 'dependents', 'education',
        'self_employed', 'applicantincome', 'coapplicantincome',
        'loan_amount_term', 'credit_history', 'property_area', 'loan_status'
    ]
    
    # Simulated raw correlations with loanamount
    simulated_raw_corrs = {
        'loan_id': 0.1060,
        'gender': 0.1768,
        'married': 0.1935,
        'dependents': 0.3103,
        'education': 0.0258,
        'self_employed': 0.0977,
        'applicantincome': 0.0445,
        'coapplicantincome': 0.0316,
        'loan_amount_term': 0.0386,
        'credit_history': 0.0557,
        'property_area': 0.1567,
        'loan_status': 0.0934,
    }
    
    # Apply the function logic
    print("\nStep 1: Remove Identifier Columns")
    print("-" * 70)
    removed = []
    valid = []
    for feat in sample_features:
        if is_identifier_column(feat):
            removed.append(feat)
            print(f"  REMOVED: {feat}")
        else:
            valid.append(feat)
    
    print(f"\nIdentifier columns removed: {removed}")
    print(f"Valid features remaining: {len(valid)}")
    
    # Compute absolute correlations
    print("\nStep 2: Compute Absolute Correlations")
    print("-" * 70)
    abs_corrs = {k: abs(v) for k, v in simulated_raw_corrs.items() if k not in removed}
    
    # Normalize
    max_corr = max(abs_corrs.values())
    print(f"Max absolute correlation: {max_corr:.4f}")
    
    print("\nStep 3: Normalized Importance Scores")
    print("-" * 70)
    print(f"{'Rank':<5} {'Feature':<20} {'Raw Corr':<12} {'Abs Corr':<12} {'Normalized %':<12}")
    print("-" * 70)
    
    normalized = {}
    for feat in valid:
        raw = simulated_raw_corrs[feat]
        abs_val = abs_corrs[feat]
        norm = (abs_val / max_corr) * 100
        normalized[feat] = round(norm, 2)
    
    # Sort by normalized score
    sorted_items = sorted(normalized.items(), key=lambda x: x[1], reverse=True)
    for rank, (feat, norm) in enumerate(sorted_items, 1):
        raw = simulated_raw_corrs[feat]
        abs_val = abs_corrs[feat]
        print(f"{rank:<5} {feat:<20} {raw:>+.4f}     {abs_val:.4f}      {norm:.1f}%")
    
    print("=" * 70)
    print("RESULT: Top feature (dependents) shows 100%, others relative to it")
    print("=" * 70)

