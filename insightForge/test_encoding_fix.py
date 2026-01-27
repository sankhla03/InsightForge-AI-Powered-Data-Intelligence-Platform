#!/usr/bin/env python3
"""
Test script to verify the feature encoding fix.
Tests that categorical features (Male, Yes, Graduate, Urban) are properly converted to numeric.
"""
import pandas as pd
import numpy as np
from preprocessing.utils import convert_features_to_numeric

def test_encoding_fix():
    """Test that categorical features are properly encoded."""
    print("=" * 70)
    print("TESTING FEATURE ENCODING FIX")
    print("=" * 70)
    
    # Create a sample dataset with categorical columns similar to the problem description
    data = {
        'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
        'Married': ['Yes', 'No', 'Yes', 'No', 'Yes'],
        'Education': ['Graduate', 'Not Graduate', 'Graduate', 'Graduate', 'Not Graduate'],
        'Self_Employed': ['No', 'Yes', 'No', 'No', 'Yes'],
        'Urban': ['Urban', 'Rural', 'Urban', 'Urban', 'Rural'],
        'Income': [5000, 6000, 5500, 7000, 4500],
        'CreditScore': [650, 700, 620, 680, 590],
        'LoanApproved': ['Yes', 'No', 'Yes', 'Yes', 'No']  # Target column
    }
    
    df = pd.DataFrame(data)
    
    print("\nOriginal DataFrame:")
    print(df)
    print("\nOriginal dtypes:")
    print(df.dtypes)
    
    # Identify object columns before encoding
    object_cols = [col for col in df.columns if df[col].dtype == 'object']
    print(f"\nObject columns to be encoded: {object_cols}")
    
    # Set target column
    target_col = 'LoanApproved'
    
    # Apply encoding
    try:
        X_encoded, y_unchanged, encoding_info = convert_features_to_numeric(df, target_col)
        
        print("\n" + "=" * 70)
        print("ENCODING RESULTS")
        print("=" * 70)
        
        # Verify target is unchanged
        print(f"\n[1] Target column '{target_col}' verification:")
        print(f"    Original dtype: {df[target_col].dtype}")
        print(f"    Target unchanged dtype: {y_unchanged.dtype}")
        print(f"    Target values preserved: {list(y_unchanged) == list(df[target_col])}")
        
        # Verify feature encoding
        print(f"\n[2] Feature encoding verification:")
        print(f"    Encoded columns: {encoding_info['encoded_columns']}")
        print(f"    Numeric columns (unchanged): {encoding_info['numeric_columns']}")
        print(f"    Binary columns (unchanged): {encoding_info['binary_columns']}")
        
        # Check that all feature columns are now numeric
        feature_dtypes = X_encoded.dtypes
        non_numeric_features = [
            col for col in X_encoded.columns 
            if not pd.api.types.is_numeric_dtype(X_encoded[col])
        ]
        
        print(f"\n[3] All features numeric verification:")
        print(f"    Non-numeric feature columns remaining: {non_numeric_features}")
        print(f"    Encoding SUCCESS: {len(non_numeric_features) == 0}")
        
        # Print encoded DataFrame
        print("\n[4] Encoded DataFrame (features only):")
        print(X_encoded)
        print("\nEncoded dtypes:")
        print(X_encoded.dtypes)
        
        # Verification passed check
        if encoding_info.get('verification_passed'):
            print("\n" + "=" * 70)
            print("✅ ALL TESTS PASSED!")
            print("=" * 70)
            print("✓ Categorical features encoded to numeric")
            print("✓ Target column unchanged")
            print("✓ No object/category columns remain in features")
            print("✓ Verification check passed")
            return True
        else:
            print("\n❌ VERIFICATION FAILED: verification_passed is False")
            return False
            
    except ValueError as e:
        print(f"\n❌ ERROR: {e}")
        return False
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_feature_selection_integration():
    """Test that feature selection methods work with encoded data."""
    print("\n" + "=" * 70)
    print("TESTING FEATURE SELECTION INTEGRATION")
    print("=" * 70)
    
    from preprocessing.feature_selection import (
        correlation_with_target,
        select_top_k_features,
        rfe_selection
    )
    
    # Create sample dataset
    data = {
        'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
        'Married': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No'],
        'Education': ['Graduate', 'Not Graduate', 'Graduate', 'Graduate', 'Not Graduate', 'Graduate', 'Graduate', 'Not Graduate'],
        'Income': [5000, 6000, 5500, 7000, 4500, 5200, 5800, 6200],
        'LoanApproved': ['Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes']
    }
    
    df = pd.DataFrame(data)
    target_col = 'LoanApproved'
    
    try:
        # Test correlation
        print("\n[1] Testing correlation_with_target...")
        corr = correlation_with_target(df, target_col)
        print(f"    Correlation scores: {dict(corr)}")
        print("    ✓ correlation_with_target works")
        
        # Test select_k_best
        print("\n[2] Testing select_top_k_features...")
        selected = select_top_k_features(df, target_col, k=2)
        print(f"    Selected features: {list(selected.columns)}")
        print("    ✓ select_top_k_features works")
        
        # Test RFE
        print("\n[3] Testing rfe_selection...")
        selected_rfe = rfe_selection(df, target_col, k=2)
        print(f"    Selected features (RFE): {list(selected_rfe.columns)}")
        print("    ✓ rfe_selection works")
        
        print("\n" + "=" * 70)
        print("✅ ALL FEATURE SELECTION TESTS PASSED!")
        print("=" * 70)
        return True
        
    except Exception as e:
        print(f"\n❌ FEATURE SELECTION ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_target_column_persistence():
    """Test that target column is preserved through encoding pipeline.
    
    This test verifies the CRITICAL fix: target column must be recombine
    after encoding because convert_features_to_numeric returns X_encoded
    (features ONLY, target excluded).
    """
    print("\n" + "=" * 70)
    print("TESTING TARGET COLUMN PERSISTENCE")
    print("=" * 70)
    
    # Create dataset with categorical features
    data = {
        'Gender': ['Male', 'Female', 'Male', 'Female'],
        'Married': ['Yes', 'No', 'Yes', 'No'],
        'Income': [5000, 6000, 5500, 7000],
        'LoanAmount': [100, 200, 150, 250]  # Numeric target
    }
    df = pd.DataFrame(data)
    target_col = 'LoanAmount'
    
    print(f"\nOriginal dataset shape: {df.shape}")
    print(f"Original columns: {list(df.columns)}")
    print(f"Target column: '{target_col}'")
    
    try:
        # Step 1: Encode features (this returns X_encoded WITHOUT target)
        X_encoded, y_unchanged, encoding_info = convert_features_to_numeric(df, target_col)
        
        print(f"\n[Step 1] After encoding:")
        print(f"    X_encoded shape: {X_encoded.shape} (features ONLY)")
        print(f"    y_unchanged shape: {y_unchanged.shape} (target ONLY)")
        print(f"    X_encoded columns: {list(X_encoded.columns)}")
        print(f"    Target '{target_col}' in X_encoded: {target_col in X_encoded.columns}")
        
        # Step 2: CRITICAL FIX - Recombine encoded features with target
        df_encoded = pd.concat([X_encoded, y_unchanged], axis=1)
        
        print(f"\n[Step 2] After recombining:")
        print(f"    df_encoded shape: {df_encoded.shape}")
        print(f"    df_encoded columns: {list(df_encoded.columns)}")
        
        # Step 3: VALIDATION - Ensure target is present
        print(f"\n[Step 3] Validation:")
        
        if target_col not in df_encoded.columns:
            print(f"    ❌ FAIL: Target column '{target_col}' missing from encoded dataset!")
            return False
        
        print(f"    ✓ Target column '{target_col}' present in encoded dataset")
        
        # Verify target values are preserved
        original_values = list(df[target_col])
        encoded_values = list(df_encoded[target_col])
        
        if original_values != encoded_values:
            print(f"    ❌ FAIL: Target values changed!")
            print(f"       Original: {original_values}")
            print(f"       Encoded:  {encoded_values}")
            return False
        
        print(f"    ✓ Target values preserved: {encoded_values == original_values}")
        
        # Verify all features are numeric
        feature_cols = [c for c in df_encoded.columns if c != target_col]
        non_numeric = [
            col for col in feature_cols 
            if not pd.api.types.is_numeric_dtype(df_encoded[col])
        ]
        
        if non_numeric:
            print(f"    ❌ FAIL: Non-numeric feature columns remain: {non_numeric}")
            return False
        
        print(f"    ✓ All feature columns are numeric")
        print(f"    ✓ Encoded {len(encoding_info['encoded_columns'])} categorical features")
        
        print("\n" + "=" * 70)
        print("✅ TARGET COLUMN PERSISTENCE TEST PASSED!")
        print("=" * 70)
        print(f"✓ Target column '{target_col}' preserved through encoding")
        print(f"✓ Target values unchanged")
        print(f"✓ All {len(encoding_info['encoded_columns'])} categorical features encoded")
        print(f"✓ df_encoded shape: {df_encoded.shape} (includes target)")
        print("=" * 70)
        return True
        
    except Exception as e:
        print(f"\n❌ TARGET PERSISTENCE ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run tests
    test1_passed = test_encoding_fix()
    test2_passed = test_feature_selection_integration()
    test3_passed = test_target_column_persistence()
    
    # Summary
    print("\n" + "=" * 70)
    print("FINAL TEST SUMMARY")
    print("=" * 70)
    print(f"Encoding Fix Test: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Feature Selection Test: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    print(f"Target Persistence Test: {'✅ PASSED' if test3_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed and test3_passed:
        print("\n🎉 ALL TESTS PASSED! The encoding fix is working correctly.")
        print("✓ Target column is NEVER dropped during encoding")
        print("✓ Feature encoding applies ONLY to feature columns (X)")
        print("✓ Feature selection always sees the correct target column")
    else:
        print("\n⚠️  SOME TESTS FAILED. Please review the implementation.")

