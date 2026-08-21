"""
Feature Scaling Module for ML Pipeline

IMPORTANT: Feature scaling is applied AFTER feature selection.

Why Scale After Feature Selection?
====================================
1. Feature selection identifies the most relevant features for prediction
2. Scaling selected features ensures:
   - No information leakage from removed features
   - More efficient computation (only scale what matters)
   - Better model convergence for distance/gradient-based algorithms
   - Consistent feature scales for fair comparison

Pipeline Order:
===============
Raw Data → Cleaning → Outlier Handling → Label Noise → 
Feature Selection (returns selected features) → Feature Scaling → Model Training

Scaling Methods:
================
- StandardScaler: Standardizes features (mean=0, std=1)
  Best for: Normal distribution, algorithms assuming Gaussian (SVM, Logistic Regression)
  
- MinMaxScaler: Normalizes features to [0, 1] range
  Best for: Neural networks, image processing, algorithms with bounded activation

- RobustScaler: Uses median/IQR (robust to outliers)
  Best for: Data with outliers, when you want to ignore extreme values

This module:
- Scales ONLY selected features (ignores target column)
- Preserves feature names in output
- Generates HTML table for UI display
- Stores scaled data in session for downstream use
"""

import pandas as pd
import numpy as np
from io import StringIO
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# ============================================================================
# SCALING METHOD SELECTION
# ============================================================================

def get_scaler(method='standard'):
    """
    Factory function to get the appropriate scaler.
    
    Args:
        method: 'standard', 'minmax', or 'robust'
    
    Returns:
        sklearn Scaler instance
    """
    scalers = {
        'standard': StandardScaler(),
        'minmax': MinMaxScaler(),
        'robust': RobustScaler(),
    }
    return scalers.get(method.lower(), StandardScaler())


def get_scaling_method_recommendation(df, selected_features, target_col):
    """
    Recommend the best scaling method based on data characteristics.
    
    Args:
        df: Original DataFrame
        selected_features: List of selected feature names
        target_col: Name of target column
    
    Returns:
        Recommended scaling method (str)
    """
    # Get feature columns only (exclude target)
    feature_cols = [c for c in selected_features if c != target_col]
    
    if not feature_cols:
        return 'standard'
    
    # Check for outliers in features
    X = df[feature_cols].select_dtypes(include=[np.number])
    if X.empty:
        return 'standard'
    
    # Count features with high skewness or outliers
    outlier_count = 0
    for col in X.columns:
        Q1 = X[col].quantile(0.25)
        Q3 = X[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((X[col] < (Q1 - 1.5 * IQR)) | (X[col] > (Q3 + 1.5 * IQR))).sum()
        if outliers / len(X) > 0.05:  # More than 5% outliers
            outlier_count += 1
    
    # If more than 30% of features have outliers, recommend RobustScaler
    if outlier_count / len(X.columns) > 0.3:
        return 'robust'
    
    # Default recommendation
    return 'standard'


# ============================================================================
# MAIN SCALING FUNCTION
# ============================================================================

def scale_selected_features(df, selected_features, target_col, method='standard'):
    """
    Apply feature scaling ONLY to selected features.
    
    Pipeline Order:
    ================
    Feature Selection → Feature Scaling (this function) → Model Prediction
    
    Args:
        df: DataFrame containing all data
        selected_features: List of feature names selected by feature selection
                           (should NOT include target column)
        target_col: Name of target column (will NOT be scaled)
        method: Scaling method ('standard', 'minmax', 'robust')
    
    Returns:
        dict containing:
            - 'scaled_df': DataFrame with scaled features (preserves feature names)
            - 'scaler': Fitted scaler for later use
            - 'feature_cols': List of scaled feature column names
            - 'scaled_html': HTML table string for UI display
            - 'scaled_data_json': JSON string for session storage
            - 'scaling_info': Dict with scaling metadata
    """
    # Validate inputs
    if df is None or selected_features is None:
        raise ValueError("DataFrame and selected_features are required")
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")
    
    # Filter to only selected features (exclude target)
    feature_cols = [c for c in selected_features if c != target_col]
    
    if not feature_cols:
        raise ValueError("No feature columns found (after excluding target)")
    
    print("\n" + "=" * 70)
    print("FEATURE SCALING (After Feature Selection)")
    print("=" * 70)
    print(f"Method: {method}")
    print(f"Target column (NOT scaled): '{target_col}'")
    print(f"Features to scale: {len(feature_cols)} columns")
    print(f"Feature names: {feature_cols}")
    print(f"Dataset shape: {df.shape}")
    print("=" * 70)
    
    # Check if features exist in dataframe
    missing_features = [f for f in feature_cols if f not in df.columns]
    if missing_features:
        raise ValueError(
            f"Selected features not found in DataFrame: {missing_features}"
        )
    
    # Get feature data (exclude target)
    X = df[feature_cols].copy()
    
    # Separate numeric and non-numeric columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = [c for c in feature_cols if c not in numeric_cols]
    
    print(f"\n[COLUMN ANALYSIS]")
    print(f"  Numeric columns (will scale): {len(numeric_cols)}")
    print(f"  Non-numeric columns (will skip): {len(non_numeric_cols)}")
    
    if non_numeric_cols:
        print(f"  Non-numeric columns: {non_numeric_cols}")
    
    # Initialize scaler
    scaler = get_scaler(method)
    
    # Scale numeric features
    X_scaled = X.copy()
    
    if numeric_cols:
        print(f"\n[SCALING {len(numeric_cols)} NUMERIC FEATURES]")
        print(f"  Scaler: {method}")
        
        # Fit and transform
        X_scaled[numeric_cols] = scaler.fit_transform(X[numeric_cols])
        
        # Log scaling statistics
        print(f"\n  Scaling Statistics:")
        print(f"  {'Feature':<25} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
        print(f"  {'-'*75}")
        
        for col in numeric_cols[:5]:  # Show first 5
            print(f"  {col:<25} {X_scaled[col].mean():>12.4f} "
                  f"{X_scaled[col].std():>12.4f} {X_scaled[col].min():>12.4f} "
                  f"{X_scaled[col].max():>12.4f}")
        
        if len(numeric_cols) > 5:
            print(f"  ... and {len(numeric_cols) - 5} more features")
    
    # Non-numeric columns remain unchanged
    if non_numeric_cols:
        print(f"\n[SKIPPING {len(non_numeric_cols)} NON-NUMERIC FEATURES]")
        for col in non_numeric_cols:
            print(f"  {col}: dtype = {X[col].dtype}")
    
    # Add target column back (unscaled)
    X_scaled[target_col] = df[target_col].values
    
    # Generate HTML table for UI display
    scaled_html = X_scaled.head(20).to_html(
        classes="table table-bordered table-striped",
        index=False,
        float_format=lambda x: f'{x:.4f}' if isinstance(x, float) else x
    )
    
    # Generate JSON for session storage
    scaled_data_json = X_scaled.to_json(orient="columns")
    
    # Create scaling info dict
    scaling_info = {
        'method': method,
        'target_column': target_col,
        'feature_columns': feature_cols,
        'numeric_features_scaled': numeric_cols,
        'non_numeric_features_skipped': non_numeric_cols,
        'n_samples': len(X_scaled),
        'n_features': len(feature_cols),
        'scaler_params': getattr(scaler, 'with_mean', None) if hasattr(scaler, 'with_mean') else None,
    }
    
    print(f"\n[OUTPUT SUMMARY]")
    print(f"  Scaled dataset shape: {X_scaled.shape}")
    print(f"  Features scaled: {len(numeric_cols)}")
    print(f"  Features skipped: {len(non_numeric_cols)}")
    print(f"  Target column preserved: '{target_col}'")
    print("=" * 70)
    
    return {
        'scaled_df': X_scaled,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'scaled_html': scaled_html,
        'scaled_data_json': scaled_data_json,
        'scaling_info': scaling_info,
    }


# ============================================================================
# APPLY SCALING TO NEW DATA (FOR PREDICTION)
# ============================================================================

def apply_scaling_to_new_data(new_data, feature_cols, scaler):
    """
    Apply fitted scaler to new data for prediction.
    
    Args:
        new_data: DataFrame or array-like with feature values
        feature_cols: List of feature column names
        scaler: Fitted sklearn scaler
    
    Returns:
        Scaled data array
    """
    if isinstance(new_data, dict):
        new_data = pd.DataFrame([new_data])
    
    # Extract only the feature columns (in correct order)
    X_new = new_data[feature_cols].select_dtypes(include=[np.number])
    
    # Apply scaling
    return scaler.transform(X_new)


# ============================================================================
# DISPLAY FUNCTIONS
# ============================================================================

def get_scaling_summary(scaled_result):
    """
    Generate a summary report of the scaling operation.
    
    Args:
        scaled_result: Output from scale_selected_features()
    
    Returns:
        Formatted summary string
    """
    info = scaled_result['scaling_info']
    
    summary = f"""
╔══════════════════════════════════════════════════════════════════════╗
║                      FEATURE SCALING SUMMARY                         ║
╠══════════════════════════════════════════════════════════════════════╣
║  Method:           {info['method']:<15}                              ║
║  Target Column:    {info['target_column']:<15} (NOT scaled)          ║
║  Features Scaled:  {info['n_features']:<15} numeric features         ║
║  Features Skipped: {len(info['non_numeric_features_skipped']):<15} non-numeric           ║
║  Samples:          {info['n_samples']:<15}                           ║
╚══════════════════════════════════════════════════════════════════════╝
    """.strip()
    
    return summary


def get_scaled_table_html(scaled_df, max_rows=20):
    """
    Generate HTML table for displaying scaled data.
    
    Args:
        scaled_df: Scaled DataFrame
        max_rows: Maximum number of rows to display
    
    Returns:
        HTML table string
    """
    display_df = scaled_df.head(max_rows)
    
    return display_df.to_html(
        classes="table table-bordered table-striped",
        index=False,
        float_format=lambda x: f'{x:.4f}' if isinstance(x, float) else x
    )


# ============================================================================
# INTEGRATION WITH MODEL TRAINING
# ============================================================================

def scale_for_model_training(df, selected_features, target_col, method='standard'):
    """
    Scale features for model training with additional outputs for training pipeline.
    
    Args:
        df: DataFrame with all data
        selected_features: Features selected by feature selection
        target_col: Target column name
        method: Scaling method
    
    Returns:
        tuple: (X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols)
    """
    from sklearn.model_selection import train_test_split
    
    # Scale features
    result = scale_selected_features(df, selected_features, target_col, method)
    
    # Split into train/test (80/20 split)
    X = result['scaled_df'].drop(columns=[target_col])
    y = result['scaled_df'][target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("\n[TRAIN-TEST SPLIT]")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Features: {X_train.shape[1]}")
    
    return (
        X_train, X_test, y_train, y_test,
        result['scaler'],
        result['feature_cols'],
        result['scaled_html']
    )


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example demonstrating the scaling pipeline
    
    # Sample data
    sample_data = {
        'feature_1': [100, 200, 300, 400, 500],
        'feature_2': [1.0, 2.0, 3.0, 4.0, 5.0],
        'feature_3': [10, 20, 30, 40, 50],  # Will be removed by feature selection
        'category': ['A', 'B', 'C', 'A', 'B'],  # Non-numeric (won't be scaled)
        'target': [0, 1, 0, 1, 0]
    }
    
    df = pd.DataFrame(sample_data)
    
    # Simulate feature selection output
    selected_features = ['feature_1', 'feature_2', 'category', 'target']
    target_col = 'target'
    
    print("\n" + "=" * 70)
    print("EXAMPLE: Feature Scaling After Feature Selection")
    print("=" * 70)
    
    # Apply scaling
    result = scale_selected_features(
        df=df,
        selected_features=selected_features,
        target_col=target_col,
        method='standard'
    )
    
    # Show results
    print("\n[Scalable Data Preview]")
    print(result['scaled_df'].head())
    
    print("\n[Scaling Info]")
    print(result['scaling_info'])

