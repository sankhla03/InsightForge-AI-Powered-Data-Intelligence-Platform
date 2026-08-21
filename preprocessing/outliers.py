import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def detect_outliers_iqr(df, multiplier=1.5, exclude_columns=None):
    """
    Detect outliers using IQR method.
    
    IMPORTANT: This function STRICTLY AVOIDS detecting outliers in:
    - Binary features (0/1) - nunique <= 2
    - Low-cardinality features - nunique <= 10
    - Non-numeric columns
    
    Args:
        df: Input DataFrame
        multiplier: IQR multiplier for bounds (default 1.5)
        exclude_columns: List of columns to exclude (e.g., target column)
    
    Returns:
        outlier_count (int)
        outlier_indices (set)
        outlier_details (dict): Maps index to list of outlier info
    """
    outlier_indices = set()
    outlier_details = {}  # {row_index: [{'column': col, 'value': val, 'lower': lower, 'upper': upper}, ...]}

    # Get feature types to identify which columns to process
    feature_types = identify_feature_types(df, target_column=exclude_columns)
    
    # Only process continuous numeric columns
    continuous_cols = feature_types['continuous']

    for col in continuous_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        # Skip if IQR is zero
        if IQR == 0:
            continue

        lower = Q1 - multiplier * IQR
        upper = Q3 + multiplier * IQR

        # Find outliers for this column
        col_outliers = df[(df[col] < lower) | (df[col] > upper)]
        
        for idx in col_outliers.index:
            outlier_indices.add(idx)
            if idx not in outlier_details:
                outlier_details[idx] = []
            outlier_details[idx].append({
                'column': col,
                'value': df.loc[idx, col],
                'lower': lower,
                'upper': upper
            })

    return len(outlier_indices), outlier_indices, outlier_details


def identify_feature_types(df, target_column=None):
    """
    Identify feature types in the dataset.
    
    Args:
        df: Input DataFrame
        target_column: Name of target column to exclude (optional)
    
    Returns:
        dict with keys:
        - 'binary': list of binary column names (nunique <= 2)
        - 'low_cardinality': list of low-cardinality columns (nunique 3-10, for numeric)
        - 'continuous': list of continuous numeric columns (nunique > 2)
        - 'non_numeric': list of non-numeric columns
    """
    feature_types = {
        'binary': [],
        'low_cardinality': [],
        'continuous': [],
        'non_numeric': []
    }
    
    for col in df.columns:
        # Skip target column
        if target_column and col == target_column:
            continue
            
        # Check if column is numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            nunique = df[col].nunique()
            
            if nunique <= 2:
                feature_types['binary'].append(col)
            elif nunique <= 10:
                feature_types['low_cardinality'].append(col)
            else:
                feature_types['continuous'].append(col)
        else:
            feature_types['non_numeric'].append(col)
    
    return feature_types


def cap_outliers_inplace(df, multiplier=1.5, target_column=None, verbose=True):
    """
    Apply IQR-based outlier capping (Winsorization) IN-PLACE to continuous features only.
    
    CRITICAL: This function updates the ORIGINAL dataset by capping only outlier values.
    - Only rows containing outliers are modified
    - Only the affected feature columns are modified
    - All non-outlier rows remain completely unchanged
    - Total row count and order are preserved
    
    IMPORTANT: This function STRICTLY AVOIDS:
    - Binary features (0/1) - capping would collapse them to single values
    - Low-cardinality features - these are categorical, not continuous
    - Features with IQR == 0 - no meaningful outliers to cap
    
    Args:
        df: Input DataFrame (will be modified in-place)
        multiplier: IQR multiplier for bounds (default 1.5, typically 1.5 or 3)
        target_column: Name of target column to exclude from outlier handling
        verbose: Whether to log progress
    
    Returns:
        outlier_indices (set): Set of row indices that were modified
        processing_report (dict): Details about what was processed
    """
    # Track which rows have outliers
    outlier_indices = set()
    
    # Identify feature types
    feature_types = identify_feature_types(df, target_column)
    
    if verbose:
        logger.info("=" * 60)
        logger.info("IN-PLACE IQR-BASED OUTLIER CAPPING")
        logger.info("=" * 60)
        logger.info(f"Target column excluded: {target_column or 'None'}")
        logger.info(f"Binary features (SKIPPED): {feature_types['binary']}")
        logger.info(f"Low-cardinality features (SKIPPED): {feature_types['low_cardinality']}")
        logger.info(f"Non-numeric features (SKIPPED): {feature_types['non_numeric']}")
        logger.info(f"Continuous features (TO PROCESS): {feature_types['continuous']}")
        logger.info(f"Dataset rows: {len(df)}")
        logger.info("-" * 60)
    
    # Track processing details
    processing_report = {
        'binary': feature_types['binary'],
        'low_cardinality': feature_types['low_cardinality'],
        'non_numeric': feature_types['non_numeric'],
        'processed': [],
        'skipped_iqr_zero': [],
        'total_capped_count': 0,
        'capped_details': {}  # {row_index: {column: {'original': value, 'capped': value}}}
    }
    
    capped_details = {}
    
    for col in feature_types['continuous']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Skip if IQR is zero (no spread in data)
        if IQR == 0:
            if verbose:
                logger.info(f"  SKIP: {col} (IQR = 0, no outliers possible)")
            processing_report['skipped_iqr_zero'].append(col)
            continue
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        # Find outlier indices for this column
        outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        col_outlier_indices = df[outlier_mask].index.tolist()
        
        # Count how many values will be capped
        below_lower = (df[col] < lower_bound).sum()
        above_upper = (df[col] > upper_bound).sum()
        total_capped = below_lower + above_upper
        
        if total_capped > 0:
            # Track capping details for each row
            for idx in col_outlier_indices:
                original_value = df.loc[idx, col]
                # Apply capping to get new value
                capped_value = max(lower_bound, min(df.loc[idx, col], upper_bound))
                
                if idx not in capped_details:
                    capped_details[idx] = {}
                capped_details[idx][col] = {
                    'original': float(original_value) if pd.notna(original_value) else None,
                    'capped': float(capped_value) if pd.notna(capped_value) else None,
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound)
                }
                outlier_indices.add(idx)
            
            # SAFE IN-PLACE UPDATE: Handle dtype conversion for integer columns
            # Convert column to float if needed to handle float bounds
            if pd.api.types.is_integer_dtype(df[col]):
                # Convert to float first, then cap, then convert back to int
                df[col] = df[col].astype(float)
                df.loc[outlier_mask, col] = df.loc[outlier_mask, col].clip(lower=lower_bound, upper=upper_bound)
                # Round to nearest integer and convert back
                df[col] = df[col].round().astype(int)
            else:
                # For float columns, just cap directly
                df.loc[outlier_mask, col] = df.loc[outlier_mask, col].clip(lower=lower_bound, upper=upper_bound)
            
            if verbose:
                logger.info(f"  PROCESS: {col}")
                logger.info(f"    Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
                logger.info(f"    Capped: {below_lower} below, {above_upper} above (total: {total_capped})")
                logger.info(f"    Rows affected: {len(set(col_outlier_indices))}")
        else:
            if verbose:
                logger.info(f"  SKIP: {col} (no outliers found)")
    
    processing_report['capped_details'] = capped_details
    processing_report['total_capped_count'] = len(outlier_indices)
    
    if verbose:
        logger.info("-" * 60)
        logger.info(f"Total rows modified: {len(outlier_indices)}")
        logger.info(f"Total values capped: {processing_report['total_capped_count']}")
        logger.info(f"Features processed: {len(processing_report['processed'])}")
        logger.info(f"Features skipped (IQR=0): {len(processing_report['skipped_iqr_zero'])}")
        logger.info(f"Dataset rows (preserved): {len(df)}")
        logger.info("=" * 60)
        logger.info("Dataset updated successfully!")
        logger.info("=" * 60)
    
    return outlier_indices, processing_report


def cap_outliers_iqr(df, multiplier=1.5, target_column=None, verbose=True):
    """
    Apply IQR-based outlier capping (Winsorization) to continuous features only.
    
    NOTE: This function creates a copy of the dataframe. For in-place updates,
    use cap_outliers_inplace() instead.
    
    IMPORTANT: This function STRICTLY AVOIDS:
    - Binary features (0/1) - capping would collapse them to single values
    - Low-cardinality features - these are categorical, not continuous
    - Features with IQR == 0 - no meaningful outliers to cap
    
    Args:
        df: Input DataFrame
        multiplier: IQR multiplier for bounds (default 1.5, typically 1.5 or 3)
        target_column: Name of target column to exclude from outlier handling
        verbose: Whether to log progress
    
    Returns:
        df_capped: DataFrame with outliers capped
        processing_report: dict with details about what was processed
    """
    df_capped = df.copy()
    
    # Identify feature types
    feature_types = identify_feature_types(df_capped, target_column)
    
    if verbose:
        logger.info("=" * 60)
        logger.info("IQR-BASED OUTLIER CAPPING (COPY)")
        logger.info("=" * 60)
        logger.info(f"Target column excluded: {target_column or 'None'}")
        logger.info(f"Binary features (SKIPPED): {feature_types['binary']}")
        logger.info(f"Low-cardinality features (SKIPPED): {feature_types['low_cardinality']}")
        logger.info(f"Non-numeric features (SKIPPED): {feature_types['non_numeric']}")
        logger.info(f"Continuous features (TO PROCESS): {feature_types['continuous']}")
    
    # Track processing details
    processing_report = {
        'binary': feature_types['binary'],
        'low_cardinality': feature_types['low_cardinality'],
        'non_numeric': feature_types['non_numeric'],
        'processed': [],
        'skipped_iqr_zero': [],
        'total_capped_count': 0
    }
    
    capped_counts = {}
    
    for col in feature_types['continuous']:
        Q1 = df_capped[col].quantile(0.25)
        Q3 = df_capped[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Skip if IQR is zero (no spread in data)
        if IQR == 0:
            if verbose:
                logger.info(f"  SKIP: {col} (IQR = 0, no outliers possible)")
            processing_report['skipped_iqr_zero'].append(col)
            continue
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        # Count how many values will be capped
        below_lower = (df_capped[col] < lower_bound).sum()
        above_upper = (df_capped[col] > upper_bound).sum()
        total_capped = below_lower + above_upper
        
        # Apply capping
        original_values = df_capped[col].copy()
        df_capped[col] = df_capped[col].clip(lower=lower_bound, upper=upper_bound)
        
        capped_counts[col] = {
            'below_lower': below_lower,
            'above_upper': above_upper,
            'total': total_capped,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
        
        processing_report['processed'].append({
            'column': col,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'values_capped': total_capped
        })
        processing_report['total_capped_count'] += total_capped
        
        if verbose:
            logger.info(f"  PROCESS: {col}")
            logger.info(f"    Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
            logger.info(f"    Capped: {below_lower} below, {above_upper} above (total: {total_capped})")
    
    if verbose:
        logger.info("-" * 60)
        logger.info(f"Total values capped: {processing_report['total_capped_count']}")
        logger.info(f"Features processed: {len(processing_report['processed'])}")
        logger.info(f"Features skipped (IQR=0): {len(processing_report['skipped_iqr_zero'])}")
        logger.info("=" * 60)
    
    return df_capped, processing_report


def remove_outliers(df, indices):
    """Remove rows containing outliers"""
    return df.drop(index=indices)


# Example usage and testing
if __name__ == "__main__":
    # Demo with sample data
    sample_data = {
        'binary_feature': [0, 1, 1, 0, 1, 0, 1, 0, 1, 0],  # SKIP - binary
        'low_card_feature': [1, 2, 3, 1, 2, 3, 1, 2, 3, 1],  # SKIP - low cardinality
        'continuous_feature1': [10, 15, 12, 100, 11, 13, -50, 14, 16, 1000],  # PROCESS - has outliers
        'continuous_feature2': [1.0, 2.0, 1.5, 1.2, 1.8, 2.1, 1.3, 1.7, 1.4, 1.6],  # PROCESS - no outliers
        'target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # SKIP - target
    }
    
    df_sample = pd.DataFrame(sample_data)
    
    print("\n" + "=" * 60)
    print("SAMPLE DATA - BEFORE OUTLIER CAPPING (IN-PLACE)")
    print("=" * 60)
    print(f"Shape: {df_sample.shape}")
    print(df_sample)
    
    # Test in-place capping
    outlier_indices, report = cap_outliers_inplace(df_sample, target_column='target')
    
    print("\n" + "=" * 60)
    print("SAMPLE DATA - AFTER IN-PLACE OUTLIER CAPPING")
    print("=" * 60)
    print(f"Shape: {df_sample.shape} (unchanged)")
    print(f"Rows modified: {len(outlier_indices)}")
    print(df_sample)
    
    print("\n" + "=" * 60)
    print("CAPPING DETAILS")
    print("=" * 60)
    for idx, details in report['capped_details'].items():
        print(f"\nRow {idx}:")
        for col, info in details.items():
            print(f"  {col}: {info['original']:.2f} -> {info['capped']:.2f} (bounds: [{info['lower_bound']:.2f}, {info['upper_bound']:.2f}])")
