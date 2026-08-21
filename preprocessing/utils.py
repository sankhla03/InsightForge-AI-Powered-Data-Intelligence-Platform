import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
import logging

# Configure logging for consistent output
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def convert_features_to_numeric(df, target_col):
    """
    Convert all FEATURE columns to numeric while preserving the TARGET column unchanged.
    
    This function implements the critical requirement of separating features (X) from 
    the target (y) BEFORE any transformations. The target column is explicitly excluded
    from ALL encoders to prevent data leakage and maintain target integrity.
    
    Pipeline Order:
        Target Selection → Feature / Target Split (X, y) → Ordinal Encoding (FEATURES ONLY)
        → Outlier Handling → Feature Selection → Feature Scaling → Model Training
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame containing both features and target column
    target_col : str
        Name of the target column to EXCLUDE from encoding
        
    Returns:
    --------
    tuple: (X_encoded, y_unchanged, encoding_info)
        X_encoded : pd.DataFrame
            Features with categorical columns encoded to numeric (int64/float64)
        y_unchanged : pd.Series
            Target column with original dtype preserved, unchanged
        encoding_info : dict
            Details about which columns were encoded, unchanged, etc.
            
    Raises:
    -------
    ValueError
        If target_col is missing from df
        If any object/category columns remain in X after encoding (verification failure)
        
    Encoding Rules:
    ---------------
    - Already numeric columns → unchanged
    - Object/categorical columns → OrdinalEncoder applied
    - Binary numeric (0/1) → unchanged (detected by nunique <= 2)
    - Target column → NEVER touched
    
    Encoder Configuration (MANDATORY):
    -----------------------------------
    OrdinalEncoder(
        handle_unknown='use_encoded_value',
        unknown_value=-1
    )
    
    Verification (MANDATORY):
    --------------------------
    - Print/log dtypes BEFORE encoding
    - Print/log dtypes AFTER encoding
    - If any object column still exists → raise error
    """
    # =========================================================================
    # STEP 1: VALIDATION - Ensure target column exists
    # =========================================================================
    if target_col not in df.columns:
        error_msg = (
            f"TARGET COLUMN '{target_col}' NOT FOUND in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info("=" * 70)
    logger.info("FEATURE TO NUMERIC CONVERSION")
    logger.info("=" * 70)
    logger.info(f"Target column (EXCLUDED): '{target_col}'")
    logger.info(f"Original dataset shape: {df.shape}")
    
    # =========================================================================
    # VERIFICATION STEP: Print dtypes BEFORE encoding
    # =========================================================================
    logger.info(f"\n[VERIFICATION] DTYPES BEFORE ENCODING:")
    logger.info("-" * 50)
    for col in df.columns:
        logger.info(f"  {col}: {df[col].dtype}")
    
    # Identify object/category/string columns BEFORE encoding
    # MANDATORY: Detect categorical feature columns using dtype: include=['object', 'category']
    # Also handle pandas StringDtype and Python str dtype
    def is_string_dtype(dtype):
        """Check if dtype is a string type (object, string, or str)"""
        dtype_str = str(dtype)
        return (
            pd.api.types.is_object_dtype(dtype) or 
            pd.api.types.is_categorical_dtype(dtype) or
            dtype_str == 'string' or
            dtype_str == 'str'
        )
    
    object_cols_before = [col for col in df.columns if is_string_dtype(df[col].dtype)]
    logger.info(f"\n[VERIFICATION] Object/Category columns found: {object_cols_before}")
    
    # =========================================================================
    # STEP 2: EXPLICIT SPLIT - Separate features from target
    # =========================================================================
    # CRITICAL: Target is extracted FIRST and stored separately
    # This ensures target is NEVER seen by any encoder
    # MANDATORY REQUIREMENT: Target column MUST be selected FIRST
    y_unchanged = df[target_col].copy()
    
    # Features = all columns EXCEPT target
    # MANDATORY REQUIREMENT: Split dataset explicitly
    X_features = df.drop(columns=[target_col])
    
    logger.info(f"\nAfter split:")
    logger.info(f"  X (features) shape: {X_features.shape}")
    logger.info(f"  y (target) shape: {y_unchanged.shape}")
    logger.info(f"  y dtype preserved: {y_unchanged.dtype}")
    
    # =========================================================================
    # STEP 3: IDENTIFY COLUMN TYPES - Detect categorical feature columns
    # =========================================================================
    # MANDATORY REQUIREMENT: Detect categorical feature columns using dtype
    # include=['object', 'category']
    
    # Helper function to detect string/categorical types
    def is_string_dtype(dtype):
        """Check if dtype is a string type (object, string, or str)"""
        dtype_str = str(dtype)
        return (
            pd.api.types.is_object_dtype(dtype) or 
            pd.api.types.is_categorical_dtype(dtype) or
            dtype_str == 'string' or
            dtype_str == 'str'
        )
    
    encoded_cols = []       # Will be encoded (categorical/object)
    numeric_cols = []       # Already numeric, unchanged
    binary_cols = []        # Binary numeric (0/1), unchanged
    skipped_cols = []       # Other non-numeric, skipped
    
    for col in X_features.columns:
        col_dtype = X_features[col].dtype
        
        # MANDATORY REQUIREMENT: Binary numeric columns (0/1) must remain unchanged
        if pd.api.types.is_numeric_dtype(col_dtype):
            nunique = X_features[col].nunique()
            if nunique <= 2:
                # Binary feature (0/1) - skip encoding
                binary_cols.append(col)
            else:
                # Numeric but not binary - already suitable
                numeric_cols.append(col)
        elif is_string_dtype(col_dtype):
            # Categorical/string - needs encoding
            # MANDATORY: Detect categorical feature columns using dtype: include=['object', 'category']
            encoded_cols.append(col)
        else:
            # Other types (datetime, etc.) - skip for now
            skipped_cols.append(col)
    
    # =========================================================================
    # STEP 4: LOG COLUMN CLASSIFICATION
    # =========================================================================
    logger.info(f"\nColumn Classification:")
    logger.info(f"  Columns to ENCODE ({len(encoded_cols)}): {encoded_cols}")
    logger.info(f"  Numeric columns (unchanged, {len(numeric_cols)}): {numeric_cols}")
    logger.info(f"  Binary columns (skipped, {len(binary_cols)}): {binary_cols}")
    logger.info(f"  Skipped columns ({len(skipped_cols)}): {skipped_cols}")
    
    # =========================================================================
    # STEP 5: APPLY ORDINAL ENCODING TO CATEGORICAL FEATURES ONLY
    # =========================================================================
    X_encoded = X_features.copy()
    
    if encoded_cols:
        logger.info(f"\nApplying OrdinalEncoder to {len(encoded_cols)} categorical features...")
        
        # MANDATORY REQUIREMENT: Encoder configuration
        # MANDATORY REQUIREMENT: handle_unknown='use_encoded_value', unknown_value=-1
        encoder = OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=-1
        )
        
        # Fit and transform ONLY the categorical columns (FEATURES ONLY, NOT TARGET)
        # MANDATORY REQUIREMENT: Explicit reassignment of encoded values
        X_encoded[encoded_cols] = encoder.fit_transform(X_features[encoded_cols].astype(str))
        
        # Log encoding results
        logger.info(f"Encoding complete. Updated dtypes:")
        for col in encoded_cols:
            logger.info(f"  {col}: {X_features[col].dtype} → {X_encoded[col].dtype}")
    else:
        logger.info("\nNo categorical columns to encode.")
    
    # =========================================================================
    # STEP 6: VERIFICATION - Check that encoding truly occurred
    # =========================================================================
    logger.info(f"\n[VERIFICATION] DTYPES AFTER ENCODING:")
    logger.info("-" * 50)
    for col in X_encoded.columns:
        logger.info(f"  {col}: {X_encoded[col].dtype}")
    
    # MANDATORY VERIFICATION: Check if any object/category/string columns remain in X
    # Reuse the is_string_dtype helper function
    remaining_object_cols = [
        col for col in X_encoded.columns 
        if is_string_dtype(X_encoded[col].dtype)
    ]
    
    # MANDATORY REQUIREMENT: If any object column still exists → raise error
    if remaining_object_cols:
        error_msg = (
            f"ENCODING VERIFICATION FAILED! "
            f"Object/Category columns still exist in X after encoding: {remaining_object_cols}. "
            f"Expected all feature columns to be numeric after encoding."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info(f"\n[VERIFICATION] SUCCESS: No object/category columns remain in X.")
    logger.info(f"  All {len(encoded_cols)} categorical feature columns encoded to numeric.")
    
    # =========================================================================
    # STEP 7: FINAL LOGGING AND SUMMARY
    # =========================================================================
    logger.info(f"\n{'=' * 70}")
    logger.info("CONVERSION SUMMARY")
    logger.info(f"{'=' * 70}")
    logger.info(f"Target column '{target_col}' - UNCHANGED (dtype: {y_unchanged.dtype})")
    logger.info(f"Features encoded: {len(encoded_cols)} columns ({encoded_cols})")
    logger.info(f"Features unchanged: {len(numeric_cols) + len(binary_cols)} columns")
    logger.info(f"  - Numeric columns: {len(numeric_cols)}")
    logger.info(f"  - Binary columns: {len(binary_cols)}")
    logger.info(f"Final X shape: {X_encoded.shape}")
    logger.info(f"Final X dtypes:\n{X_encoded.dtypes.to_string()}")
    logger.info(f"Final y shape: {y_unchanged.shape}")
    logger.info(f"{'=' * 70}")
    
    # =========================================================================
    # STEP 8: RETURN - X encoded, y unchanged
    # =========================================================================
    # Build encoding info dict for consistency
    encoding_info = {
        'encoded_columns': encoded_cols,
        'numeric_columns': numeric_cols,
        'binary_columns': binary_cols,
        'skipped_columns': skipped_cols,
        'target_column': target_col,
        'target_dtype_preserved': str(y_unchanged.dtype),
        'verification_passed': True,
        'original_object_cols': object_cols_before,
        'remaining_object_cols': remaining_object_cols
    }
    
    # MANDATORY REQUIREMENT: UI confirmation should be shown ONLY AFTER successful conversion
    # This is handled by the caller checking for 'verification_passed' in encoding_info
    
    return X_encoded, y_unchanged, encoding_info


def detect_irrelevant_features(df, target, corr_threshold=0.05):
    """
    Detect irrelevant features based on correlation with target.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    target : str
        Target column name
    corr_threshold : float
        Correlation threshold below which features are considered irrelevant
        
    Returns:
    --------
    list : List of irrelevant feature names
    """
    numeric_df = df.select_dtypes(include=["number"])

    # Remove constant features
    constant_features = [
        col for col in numeric_df.columns
        if numeric_df[col].nunique() <= 1
    ]

    # Correlation-based filtering
    corr = numeric_df.corr()[target].abs()
    low_corr_features = corr[corr < corr_threshold].index.tolist()

    irrelevant = set(constant_features + low_corr_features)
    irrelevant.discard(target)

    return list(irrelevant)

