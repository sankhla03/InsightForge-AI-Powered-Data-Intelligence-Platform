# preprocessing/views.py

import pandas as pd
import json
import numpy as np
import os
import joblib
from io import StringIO
from collections import Counter
from django.conf import settings

from django.shortcuts import render, redirect
from django.contrib import messages
from django.http import HttpResponse

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from .utils import detect_irrelevant_features
from .outliers import detect_outliers_iqr, cap_outliers_iqr, cap_outliers_inplace, remove_outliers
from .utils import convert_features_to_numeric
from .feature_selection import correlation_with_target

# ============================================================================
# FILE-BASED STORAGE FOR LARGE DATASETS AND SCALERS
# ============================================================================

DATA_STORAGE_DIR = os.path.join(settings.BASE_DIR, 'saved_datasets')

# ============================================================================
# SCALER STORAGE - For persistence across server restarts
# ============================================================================

SAVED_SCALERS_DIR = os.path.join(settings.BASE_DIR, 'saved_scalers')

def ensure_scalers_dir():
    """Create scaler storage directory if it doesn't exist"""
    os.makedirs(SAVED_SCALERS_DIR, exist_ok=True)


def save_scaler_to_file(scaler, method, feature_cols):
    """
    Save the fitted scaler object to a file for persistence.
    
    This is critical for production ML pipelines:
    - The scaler MUST be saved with fit() state, not just parameters
    - This ensures prediction data is transformed using the exact same scaling
      that was applied during training
    - Supports all scaler types (StandardScaler, MinMaxScaler, RobustScaler)
    
    Args:
        scaler: Fitted sklearn scaler object
        method: Scaling method name ('standard', 'minmax', 'robust')
        feature_cols: List of feature column names (for ordering)
    
    Returns:
        Path to the saved scaler file
    """
    ensure_scalers_dir()
    
    # Create unique filename with method and timestamp
    import time
    timestamp = int(time.time())
    scaler_filename = f"scaler_{method}_{timestamp}.pkl"
    scaler_path = os.path.join(SAVED_SCALERS_DIR, scaler_filename)
    
    # Save the complete scaler object (with fitted state)
    # This preserves all parameters learned during fit()
    joblib.dump(scaler, scaler_path)
    
    print("\n" + "=" * 70)
    print("SCALER SAVED TO FILE (Production-Ready)")
    print("=" * 70)
    print(f"  Scaler path: {scaler_path}")
    print(f"  Method: {method}")
    print(f"  Features: {len(feature_cols)} columns")
    print(f"  Feature order: {feature_cols}")
    print("=" * 70 + "\n")
    
    return scaler_path


def get_latest_scaler_path():
    """
    Get the path to the most recently saved scaler file.
    
    Returns:
        Path to the scaler file, or None if no scaler exists
    """
    ensure_scalers_dir()
    
    # Find all scaler files
    scaler_files = [f for f in os.listdir(SAVED_SCALERS_DIR) if f.startswith("scaler_") and f.endswith(".pkl")]
    
    if not scaler_files:
        return None
    
    # Sort by modification time, return most recent
    scaler_files.sort(key=lambda f: os.path.getmtime(os.path.join(SAVED_SCALERS_DIR, f)), reverse=True)
    return os.path.join(SAVED_SCALERS_DIR, scaler_files[0])

def ensure_storage_dir():
    """Create storage directory if it doesn't exist"""
    os.makedirs(DATA_STORAGE_DIR, exist_ok=True)

def save_dataset_to_file(df, key):
    """
    Save DataFrame to a file for large datasets.
    Returns the file path, or None if small enough for session.
    """
    ensure_storage_dir()
    file_path = os.path.join(DATA_STORAGE_DIR, f'{key}.csv')
    df.to_csv(file_path, index=False)
    return file_path

def load_dataset_from_file(file_path):
    """Load DataFrame from file"""
    return pd.read_csv(file_path)

def get_dataset(request, session_key, file_key=None):
    """
    Get dataset from session or file storage.
    Priority: session (small data) > file (large data)
    """
    # Try session first
    if session_key in request.session:
        session_data = request.session[session_key]
        
        # Check if it's a file reference (JSON object with 'file' key)
        if isinstance(session_data, str):
            try:
                # Try to parse as JSON
                parsed = json.loads(session_data)
                if isinstance(parsed, dict) and 'file' in parsed:
                    # This is a file reference
                    file_key = parsed['file']
                    file_path = os.path.join(DATA_STORAGE_DIR, f'{file_key}.csv')
                    if os.path.exists(file_path):
                        return load_dataset_from_file(file_path)
                    return None
                else:
                    # This is actual JSON data - try records format first (row-oriented)
                    try:
                        df = pd.read_json(StringIO(session_data), orient="records")
                        if len(df) > 0:
                            return df
                    except:
                        pass
                    # Fall back to columns format
                    try:
                        df = pd.read_json(StringIO(session_data), orient="columns")
                        if len(df) > 0:
                            return df
                    except:
                        pass
            except (json.JSONDecodeError, ValueError):
                # Not JSON or invalid JSON, treat as raw JSON string
                try:
                    df = pd.read_json(StringIO(session_data), orient="records")
                    if len(df) > 0:
                        return df
                except:
                    pass
                try:
                    df = pd.read_json(StringIO(session_data), orient="columns")
                    if len(df) > 0:
                        return df
                except:
                    pass
    
    # Fall back to file storage
    if file_key:
        file_path = os.path.join(DATA_STORAGE_DIR, f'{file_key}.csv')
        if os.path.exists(file_path):
            return load_dataset_from_file(file_path)
    
    return None

def save_dataset(request, df, session_key, file_key=None):
    """
    Save DataFrame to session and optionally to file.
    For large datasets, always save to file.
    """
    # Estimate size (rough estimate: ~100 bytes per cell)
    estimated_size = len(df) * len(df.columns) * 100
    
    # For large datasets, always save to file
    if estimated_size > 500000:  # ~500KB threshold
        ensure_storage_dir()
        file_path = os.path.join(DATA_STORAGE_DIR, f'{file_key}.csv')
        df.to_csv(file_path, index=False)
        # Store just the file path in session
        request.session[session_key] = json.dumps({
            'file': file_key,
            'rows': len(df),
            'cols': len(df.columns)
        })
        return file_key
    else:
        # Small dataset, store in session
        request.session[session_key] = df.to_json(orient="columns")
        return None

# =========================================================
# 1️⃣ CLEAN DATA VIEW
# =========================================================
def clean_data_view(request):
    if "dataset" not in request.session:
        return redirect("upload_dataset")

    # Bug C3 fix: use get_dataset() to handle both small (session JSON)
    # and large (file-backed) datasets consistently.
    try:
        df = get_dataset(request, "dataset", "dataset")
        if df is None:
            return redirect("upload_dataset")
    except Exception:
        return redirect("upload_dataset")

    original_rows = len(df)
    duplicate_rows = df.duplicated().sum()
    missing_before = int(df.isnull().sum().sum())

    # Normalize column names
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    # Handle missing values
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].mean())
        else:
            mode = df[col].mode()
            df[col] = df[col].fillna(
                mode.iloc[0] if not mode.empty else "Unknown"
            )

    missing_after = int(df.isnull().sum().sum())

    # Remove duplicates
    df = df.drop_duplicates()
    cleaned_rows = len(df)
    request.session["cleaning_report"] = {
        "original_rows": int(original_rows),
        "duplicate_rows": int(duplicate_rows),
        "missing_handled": int(missing_before - missing_after),
    }
    
    # Save cleaned dataset (use file storage for large datasets)
    save_dataset(request, df, "cleaned_dataset", "cleaned_dataset")
    
    # Convert DataFrame to records for full dataset display
    full_data_json = df.to_json(orient="records")

    return render(request, "preprocessing/clean.html", {
        "original_rows": original_rows,
        "duplicate_rows": duplicate_rows,
        "cleaned_rows": cleaned_rows,
        "columns": len(df.columns),
        "missing_before": missing_before,
        "missing_handled": missing_before - missing_after,
        "full_data_json": full_data_json,
        "preview": df.head(10).to_html(classes="table table-bordered"),
    })


# =========================================================
# 2️⃣ OUTLIER DETECTION VIEW
# =========================================================
def outlier_detection_view(request):
    df = get_dataset(request, "cleaned_dataset", "cleaned_dataset")
    if df is None:
        return redirect("clean_data")

    # Detect outliers only in continuous features (binary/low-cardinality are skipped)
    outlier_count, outlier_indices, outlier_details = detect_outliers_iqr(df)

    # Extract outlier rows for display
    outlier_data_json = "[]"
    if outlier_count > 0:
        outlier_indices_list = list(outlier_indices)
        # Filter to only valid indices (within bounds)
        max_idx = len(df) - 1
        valid_indices = [idx for idx in outlier_indices_list if 0 <= idx <= max_idx]
        # Remove duplicates while preserving order
        seen = set()
        valid_indices = [x for x in valid_indices if not (x in seen or seen.add(x))]
        if valid_indices:
            outlier_rows = df.iloc[valid_indices]
        else:
            outlier_rows = pd.DataFrame()
        # Add original index to each row for matching with outlier_details
        outlier_rows_with_index = outlier_rows.reset_index()
        outlier_data_json = outlier_rows_with_index.to_json(orient="records")

    # Convert outlier_details to JSON for highlighting in template
    outlier_details_json = "{}"
    if outlier_details:
        # Convert to serializable format
        serializable_details = {}
        for idx, details in outlier_details.items():
            serializable_details[str(idx)] = [
                {
                    'column': d['column'],
                    'value': float(d['value']) if d['value'] is not None else None,
                    'lower': float(d['lower']) if d['lower'] is not None else None,
                    'upper': float(d['upper']) if d['upper'] is not None else None
                }
                for d in details
            ]
        import json
        outlier_details_json = json.dumps(serializable_details)

    if request.method == "POST":
        action = request.POST.get("action")
        
        # Handle saving capped data edits
        if action == "save_capped":
            try:
                capped_data_json = request.POST.get("capped_data")
                if capped_data_json:
                    # =====================================================================
                    # CRITICAL: Update the FULL dataset in-place, not just the subset
                    # =====================================================================
                    # Load the current outlier_free_dataset (ALL rows)
                    df_full = get_dataset(request, "outlier_free_dataset", "outlier_free_dataset")
                    if df_full is None:
                        messages.error(request, "Dataset not found. Please start over.")
                        return redirect("clean_data")
                    
                    # Parse the edited capped data (only outlier rows)
                    edited_capped_df = pd.read_json(
                        StringIO(capped_data_json),
                        orient="records"
                    )
                    
                    # Get the original outlier indices that were capped
                    # We need to reload from cleaned_dataset to get original indices
                    df_original = get_dataset(request, "cleaned_dataset", "cleaned_dataset")
                    if df_original is not None:
                        # Re-detect outliers to get indices
                        _, outlier_indices_for_update, _ = detect_outliers_iqr(df_original)
                        outlier_indices_list = sorted(list(outlier_indices_for_update))
                    else:
                        # Fallback: use row count from capped data
                        outlier_indices_list = list(range(len(edited_capped_df)))
                    
                    # =====================================================================
                    # CRITICAL: Update only the outlier rows in the FULL dataset
                    # =====================================================================
                    # This preserves:
                    # - All non-outlier rows (unchanged)
                    # - Row order (unchanged)
                    # - Total row count (unchanged)
                    # - Only modifies the outlier rows with new values
                    if len(outlier_indices_list) == len(edited_capped_df):
                        # Update using index alignment
                        for i, idx in enumerate(outlier_indices_list):
                            if idx < len(df_full):
                                for col in edited_capped_df.columns:
                                    if col in df_full.columns:
                                        val = edited_capped_df.iloc[i][col]
                                        df_full.at[idx, col] = val
                    
                    # =====================================================================
                    # CRITICAL: Save the FULL updated dataset for pipeline continuity
                    # =====================================================================
                    save_dataset(request, df_full, "outlier_free_dataset", "outlier_free_dataset")
                    
                    # Set flag to show success message
                    request.session["outlier_saved"] = True
                    request.session["outlier_save_method"] = "winsorized"
                    
                    return redirect("outlier_detection")
            except Exception as e:
                messages.error(request, f"Error saving data: Invalid data format: {str(e)}")
        
        if action == "remove":
            df = remove_outliers(df, outlier_indices)
            # Bug H5 fix: use save_dataset() to ensure consistent orient
            # and support large datasets via file storage.
            save_dataset(request, df, "outlier_free_dataset", "outlier_free_dataset")
            # Set flag to show "Dataset updated" state
            request.session["outlier_saved"] = True
            request.session["outlier_save_method"] = "removed"
            return render(request, "preprocessing/outliers.html", {
                "outlier_count": outlier_count,
                "total_rows": len(df),
                "removed": True,
                "method": "removed",
            })
        
        elif action == "winsorize":
            # =====================================================================
            # CRITICAL FIX: Use in-place capping to update ORIGINAL dataset
            # =====================================================================
            # Load the original cleaned dataset (ALL rows)
            df_original = get_dataset(request, "cleaned_dataset", "cleaned_dataset")
            if df_original is None:
                messages.error(request, "Cleaned dataset not found. Please start from data cleaning.")
                return redirect("clean_data")
            
            # Detect outliers in the original dataset
            outlier_count, outlier_indices, outlier_details = detect_outliers_iqr(df_original)
            
            if outlier_count == 0:
                messages.info(request, "No outliers detected.")
                return render(request, "preprocessing/outliers.html", {
                    "outlier_count": 0,
                    "total_rows": len(df_original),
                    "removed": False,
                })
            
            # =====================================================================
            # CRITICAL: Apply in-place capping to update ONLY outlier rows
            # =====================================================================
            # This preserves:
            # - Total row count (all rows)
            # - Row order (unchanged)
            # - All non-outlier values (unchanged)
            # - Only modifies outlier values at affected indices
            outlier_indices_modified, processing_report = cap_outliers_inplace(
                df_original, 
                target_column=None, 
                verbose=True
            )
            
            # =====================================================================
            # CRITICAL: Save the FULL updated dataset (all rows with capped values)
            # =====================================================================
            # This ensures pipeline continuity - subsequent steps use all data
            # Use file storage for large datasets and also store in session as JSON
            save_dataset(request, df_original, "outlier_free_dataset", "outlier_free_dataset")
            
            # Also store as JSON in session for immediate access (in case file read fails)
            # This ensures pipeline continuity even if file read has issues
            request.session["outlier_free_dataset"] = df_original.to_json(orient="columns")
            
            # Store outlier report for report generation
            request.session["outlier_report"] = {
                "outlier_count": outlier_count,
                "method": "winsorized",
                "rows_corrected": len(outlier_indices_modified),
            }
            request.session["outlier_count"] = outlier_count
            request.session["outlier_save_method"] = "winsorized"
            
            # Verify the dataset was saved
            dataset_check = get_dataset(request, "outlier_free_dataset", "outlier_free_dataset")
            if dataset_check is not None:
                print(f"[DEBUG] outlier_free_dataset saved successfully with {len(dataset_check)} rows")
            else:
                print(f"[WARNING] outlier_free_dataset could not be retrieved after save")
            
            # Store capping details for display (capped values vs original)
            import json
            capped_details = processing_report.get('capped_details', {})
            capped_details_json = json.dumps(capped_details)
            
            # Get the outlier rows for display (using original values from before capping)
            # Reload original dataset to get original values for comparison
            df_original_reload = get_dataset(request, "cleaned_dataset", "cleaned_dataset")
            if df_original_reload is not None and len(outlier_indices_modified) > 0:
                # Get outlier rows in the same order as they were capped
                outlier_indices_list = sorted(list(outlier_indices_modified))
                outlier_rows = df_original_reload.iloc[outlier_indices_list]
                outlier_rows_with_index = outlier_rows.reset_index()
                outlier_data_json = outlier_rows_with_index.to_json(orient="records")
                
                # Convert outlier_details to JSON for highlighting
                serializable_details = {}
                for idx, details in outlier_details.items():
                    serializable_details[str(idx)] = [
                        {
                            'column': d['column'],
                            'value': float(d['value']) if d['value'] is not None else None,
                            'lower': float(d['lower']) if d['lower'] is not None else None,
                            'upper': float(d['upper']) if d['upper'] is not None else None
                        }
                        for d in details
                    ]
                outlier_details_json = json.dumps(serializable_details)
            else:
                outlier_data_json = "[]"
                outlier_details_json = "{}"
            
            # Get capped rows from the updated dataset for display
            if len(outlier_indices_modified) > 0:
                outlier_indices_list = sorted(list(outlier_indices_modified))
                capped_rows = df_original.iloc[outlier_indices_list]
                capped_data_json = capped_rows.to_json(orient="records")
            else:
                capped_data_json = "[]"
            
            request.session["outlier_saved"] = True
            request.session["outlier_save_method"] = "winsorized"
            
            # =====================================================================
            # RENDER: Show success message and capped data (no redirect)
            # =====================================================================
            return render(request, "preprocessing/outliers.html", {
                "outlier_count": outlier_count,
                "total_rows": len(df_original),
                "removed": True,
                "method": "winsorized",
                "capped_data_json": capped_data_json,
                "capped_details_json": capped_details_json,
                "outlier_data_json": outlier_data_json,
                "outlier_details_json": outlier_details_json,
                "outlier_free_dataset_rows": len(df_original),  # All rows preserved
            })

    # Check if we need to show "Dataset updated" state
    show_removed = request.session.pop("outlier_saved", False)
    save_method = request.session.pop("outlier_save_method", None)
    
    # If showing saved state, reconstruct the capped data for display
    if show_removed and save_method == "winsorized":
        # Reload from session and get the capped outlier rows
        df_from_session = get_dataset(request, "outlier_free_dataset", "outlier_free_dataset")
        if df_from_session is None:
            df_from_session = pd.DataFrame()
        
        # Re-detect outliers in the saved data to show the capped rows
        outlier_count_new, outlier_indices_new, outlier_details_new = detect_outliers_iqr(df_from_session)
        
        # Also reload original df to get original outlier data for display
        df_original = get_dataset(request, "cleaned_dataset", "cleaned_dataset")
        if df_original is None:
            df_original = pd.DataFrame()
        
        # Prepare outlier data for display (original outlier rows)
        outlier_data_json = "[]"
        outlier_details_json = "{}"
        if outlier_count_new > 0:
            outlier_indices_list = list(outlier_indices_new)
            # Filter to only valid indices (within bounds)
            max_idx = len(df_original) - 1
            valid_indices = [idx for idx in outlier_indices_list if 0 <= idx <= max_idx]
            # Remove duplicates while preserving order
            seen = set()
            valid_indices = [x for x in valid_indices if not (x in seen or seen.add(x))]
            if valid_indices:
                outlier_rows = df_original.iloc[valid_indices]
            else:
                outlier_rows = pd.DataFrame()
            outlier_rows_with_index = outlier_rows.reset_index()
            outlier_data_json = outlier_rows_with_index.to_json(orient="records")
            
            # Convert outlier_details to JSON for highlighting
            serializable_details = {}
            for idx, details in outlier_details_new.items():
                serializable_details[str(idx)] = [
                    {
                        'column': d['column'],
                        'value': float(d['value']) if d['value'] is not None else None,
                        'lower': float(d['lower']) if d['lower'] is not None else None,
                        'upper': float(d['upper']) if d['upper'] is not None else None
                    }
                    for d in details
                ]
            outlier_details_json = json.dumps(serializable_details)
        
        if outlier_count_new > 0:
            # Filter to only valid indices (within bounds)
            max_idx_session = len(df_from_session) - 1
            valid_session_indices = [idx for idx in list(outlier_indices_new) if 0 <= idx <= max_idx_session]
            seen3 = set()
            valid_session_indices = [x for x in valid_session_indices if not (x in seen3 or seen3.add(x))]
            if valid_session_indices:
                capped_rows = df_from_session.iloc[valid_session_indices]
            else:
                capped_rows = pd.DataFrame()
            capped_data_json = capped_rows.to_json(orient="records")
            return render(request, "preprocessing/outliers.html", {
                "outlier_count": outlier_count_new,
                "total_rows": len(df_from_session),
                "removed": True,
                "method": "winsorized",
                "capped_data_json": capped_data_json,
                "outlier_data_json": outlier_data_json,
                "outlier_details_json": outlier_details_json,
            })
        else:
            return render(request, "preprocessing/outliers.html", {
                "outlier_count": 0,
                "total_rows": len(df_from_session),
                "removed": True,
                "method": "winsorized",
            })
    
    return render(request, "preprocessing/outliers.html", {
        "outlier_count": outlier_count,
        "outlier_data_json": outlier_data_json,
        "outlier_details_json": outlier_details_json,
        "total_rows": len(df),
        "removed": False,
    })


# =========================================================
# 3️⃣ LABEL NOISE DETECTION VIEW
# =========================================================
def label_noise_view(request):
    # Get dataset from session or file storage
    df = get_dataset(request, "cleaned_dataset", "cleaned_dataset")
    if df is None:
        df = get_dataset(request, "outlier_free_dataset", "outlier_free_dataset")
    if df is None:
        messages.error(request, "Please clean the dataset first.")
        return redirect("upload_dataset")

    columns = df.columns.tolist()
    noise_report = None
    noise_data_json = "[]"
    noisy_count = 0
    
    # For manual noise handling, detect noise using ML (CLASSIFICATION ONLY).
    # Bug H3 fix: do NOT run RandomForestClassifier on a regression target.
    # A continuous numeric target cannot be meaningfully classified as 'noisy'.
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if numeric_cols:
        target = numeric_cols[0]
        y = df[target]

        # Only run classifier-based noise detection for classification targets
        # (discrete labels with <= 20 unique values).
        is_classification_target = y.nunique() <= 20 and not pd.api.types.is_float_dtype(y)

        if is_classification_target and y.nunique() >= 2 and min(Counter(y).values()) >= 2:
            X = df.drop(columns=[target])
            numeric_feature_cols = X.select_dtypes(include=[np.number]).columns.tolist()

            if numeric_feature_cols:
                X_numeric = X[numeric_feature_cols]

                X_train, _, y_train, _ = train_test_split(
                    X_numeric, y, test_size=0.25, random_state=42, stratify=y
                )

                model = RandomForestClassifier(n_estimators=200, random_state=42)
                model.fit(X_train, y_train)

                df["predicted_label"] = model.predict(X_numeric)
                df["is_label_noise"] = df[target] != df["predicted_label"]

                noisy_samples = df[df["is_label_noise"]].copy()
                noisy_indices = noisy_samples.index.tolist()

                if noisy_indices:
                    noise_report = {
                        "total_rows": len(df),
                        "noisy_rows": len(noisy_indices),
                        "noise_percent": round((len(noisy_indices) / len(df)) * 100, 2),
                        "target_col": target,
                    }

                    noise_data_json = noisy_samples.drop(
                        columns=["predicted_label", "is_label_noise"]
                    ).to_json(orient="records")

                    noisy_count = len(noisy_indices)
    
    # Convert full dataset for cases where no noise detected
    full_data_json = df.to_json(orient="records")
    total_rows = len(df)

    if request.method == "POST":
        action = request.POST.get("action")

        # =====================================================
        # AUTOMATIC NOISE HANDLING
        # =====================================================
        if action == "auto_detect":
            target = request.POST.get("target")
            
            if target not in df.columns:
                messages.error(request, "Invalid target selected.")
                return redirect("label_noise")
            
            y = df[target]
            
            if y.nunique() < 2 or min(Counter(y).values()) < 2:
                messages.error(
                    request,
                    "Each class must have at least 2 samples."
                )
                return redirect("label_noise")
            
            X = df.drop(columns=[target])
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numeric_cols:
                messages.error(request, "No numeric features found.")
                return redirect("label_noise")
            
            X_numeric = X[numeric_cols]
            
            X_train, _, y_train, _ = train_test_split(
                X_numeric, y, test_size=0.25, random_state=42, stratify=y
            )
            
            model = RandomForestClassifier(n_estimators=200, random_state=42)
            model.fit(X_train, y_train)
            
            df["predicted_label"] = model.predict(X_numeric)
            df["is_label_noise"] = df[target] != df["predicted_label"]
            
            noisy_samples = df[df["is_label_noise"]].copy()
            noisy_indices = noisy_samples.index.tolist()
            
            if noisy_indices:
                noise_report = {
                    "total_rows": len(df),
                    "noisy_rows": len(noisy_indices),
                    "noise_percent": round((len(noisy_indices) / len(df)) * 100, 2),
                    "target_col": target,
                }
                
                request.session["noise_report"] = noise_report
                request.session["noisy_samples"] = noisy_samples.to_json(
                    orient="columns"
                )
                
                noise_data_json = noisy_samples.drop(
                    columns=["predicted_label", "is_label_noise"]
                ).to_json(orient="records")
                
                noisy_count = len(noisy_indices)
                
            else:
                df = df.drop(columns=["predicted_label", "is_label_noise"])
                request.session["noise_free_dataset"] = df.to_json(orient="columns")
                messages.success(request, "No label noise detected.")
                return redirect("dashboard")
        
        # =====================================================
        # AUTO HANDLE NOISE - Remove all noisy rows automatically
        # =====================================================
        elif action == "auto_handle":
            if "noisy_samples" in request.session:
                noisy_samples = pd.read_json(
                    StringIO(request.session["noisy_samples"]),
                    orient="columns"
                )
                noisy_indices = noisy_samples.index.tolist()
                
                df = df.drop(index=noisy_indices)
                df = df.reset_index(drop=True)
                
                request.session["noise_free_dataset"] = df.to_json(orient="columns")
                messages.success(request, f"Auto-handled: Removed {len(noisy_indices)} noisy rows.")
                return redirect("dashboard")
            else:
                messages.error(request, "No noisy samples found.")
        
        # =====================================================
        # MANUAL NOISE HANDLING - SAVE EDITS
        # =====================================================
        elif action == "manual_save":
            try:
                edited_data_json = request.POST.get("noise_data")
                if edited_data_json:
                    # Use StringIO to properly parse JSON string
                    edited_df = pd.read_json(
                        StringIO(edited_data_json),
                        orient="records"
                    )
                    request.session["noise_free_dataset"] = edited_df.to_json(
                        orient="columns"
                    )
                    messages.success(request, "Manual edits saved successfully.")
                    return redirect("dashboard")
            except Exception as e:
                messages.error(request, f"Error saving data: Invalid data format")
        
        # =====================================================
        # MANUAL NOISE HANDLING - REMOVE DELETED
        # =====================================================
        elif action == "manual_remove":
            try:
                deleted_indices_json = request.POST.get("deleted_indices")
                if deleted_indices_json:
                    deleted_indices = set(json.loads(deleted_indices_json))
                    df = df.drop(index=deleted_indices).reset_index(drop=True)
                    
                    request.session["noise_free_dataset"] = df.to_json(
                        orient="columns"
                    )
                    messages.success(request, "Removed noisy rows successfully.")
                    return redirect("dashboard")
            except Exception as e:
                messages.error(request, f"Error removing rows: {str(e)}")

    return render(request, "preprocessing/label_noise.html", {
        "columns": columns,
        "noise_report": noise_report,
        "noise_data_json": noise_data_json,
        "noisy_count": noisy_count,
        "full_data_json": full_data_json,
        "total_rows": total_rows,
    })

# Helper function to convert numpy types to Python native types
def convert_to_native(obj):
    """Convert numpy/pandas types to native Python types for JSON serialization"""
    import json
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    elif isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(i) for i in obj]
    return obj


# =========================================================
# ENCODE FEATURES VIEW - AJAX endpoint to encode features before selection
# =========================================================
def encode_features_view(request):
    """AJAX endpoint to encode categorical features and return the full dataset"""
    if request.method != "POST":
        return HttpResponse(status=405)
    
    # Get target from POST data
    target = request.POST.get("target")
    
    if not target:
        return HttpResponse(json.dumps({"error": "Target column is required"}), status=400, content_type="application/json")
    
    # Bug C8 fix: use get_dataset() instead of raw pd.read_json() so that
    # file-backed (large) datasets are loaded correctly.
    # NOTE: explicit is None checks — `or` raises ValueError on a non-empty DataFrame.
    df = get_dataset(request, "noise_free_dataset", "noise_free_dataset")
    if df is None:
        df = get_dataset(request, "outlier_free_dataset", "outlier_free_dataset")
    if df is None:
        df = get_dataset(request, "cleaned_dataset", "cleaned_dataset")
    if df is None:
        return HttpResponse(
            json.dumps({"error": "No dataset found. Please upload and clean a dataset first."}),
            status=400,
            content_type="application/json",
        )
    
    # Normalize target column name
    if target:
        target_normalized = target.lower().replace(" ", "_")
        matched_target = None
        for col in df.columns:
            if col.lower().replace(" ", "_") == target_normalized:
                matched_target = col
                break
        if matched_target:
            target = matched_target
    
    # Check if target exists
    if target not in df.columns:
        return HttpResponse(json.dumps({"error": f"Target column '{target}' not found in dataset"}), status=400, content_type="application/json")
    
    try:
        # Apply ordinal encoding to features (TARGET EXCLUDED)
        # IMPORTANT: convert_features_to_numeric returns (X_encoded, y_unchanged, encoding_info)
        # X_encoded is features-only (target already dropped)
        X_encoded, y_unchanged, encoding_info = convert_features_to_numeric(df, target)
        
        # =========================================================================
        # CRITICAL FIX: Recombine encoded features with target column
        # =========================================================================
        # The target column was temporarily separated for encoding.
        # We MUST reattach it to create the full dataset.
        df_display = pd.concat([X_encoded, y_unchanged], axis=1)
        
        # =========================================================================
        # MANDATORY VALIDATION: Ensure target is present after encoding
        # =========================================================================
        if target not in df_display.columns:
            error_msg = (
                f"\n{'=' * 70}\n"
                f"CRITICAL ERROR: TARGET COLUMN LOST AFTER ENCODING\n"
                f"{'=' * 70}\n"
                f"Target column: '{target}'\n"
                f"Available columns: {list(df_display.columns)}\n"
                f"\nThis indicates a bug in the encoding pipeline.\n"
                f"The target column should have been reattached after encoding.\n"
                f"{'=' * 70}\n"
            )
            print(error_msg)
            raise ValueError(
                f"TARGET COLUMN '{target}' NOT FOUND in encoded dataset.\n"
                f"Available columns: {list(df_display.columns)}\n"
                f"The target column was lost during feature encoding."
            )
        
        # =========================================================================
        # MANDATORY LOGGING: Show encoding results and target validation
        # =========================================================================
        print("\n" + "=" * 70)
        print("✅ ENCODING COMPLETE - TARGET VALIDATED")
        print("=" * 70)
        print(f"Target column (PROTECTED): '{target}'")
        print(f"Target present in dataset: {target in df_display.columns}")
        print(f"Target dtype: {df_display[target].dtype}")
        print(f"Target values preserved: {list(df_display[target]) == list(y_unchanged)}")
        print(f"Features encoded: {len(encoding_info['encoded_columns'])} columns")
        print(f"Dataset shape after encoding: {df_display.shape}")
        print("=" * 70 + "\n")
        
        # Store encoding info in session for display
        request.session["feature_encoding_applied"] = True
        request.session["encoding_info"] = encoding_info
        request.session["target_column"] = target
        
        # Store encoded dataset - will be used for feature selection and as noise_free_dataset replacement
        request.session["encoded_dataset"] = df_display.to_json(orient="columns")
        # Also update noise_free_dataset so subsequent steps use the encoded data
        request.session["noise_free_dataset"] = df_display.to_json(orient="columns")
        
        # Generate full encoded data table for display
        encoded_data_html = df_display.to_html(classes="table table-bordered", index=False)
        
        return HttpResponse(json.dumps({
            "success": True,
            "encoding_info": encoding_info,
            "encoded_data_html": encoded_data_html,
            "columns": list(df_display.columns),
            "total_rows": len(df_display),
            "message": "Dataset encoded successfully! The encoded dataset is now saved and will be used for feature selection."
        }), content_type="application/json")
        
    except ValueError as e:
        return HttpResponse(json.dumps({"error": str(e)}), status=400, content_type="application/json")
    except Exception as e:
        return HttpResponse(json.dumps({"error": f"Error encoding features: {str(e)}"}), status=500, content_type="application/json")


# =========================================================
# GET ENCODED DATA VIEW - Return encoded data for display
# =========================================================
def get_encoded_data_view(request):
    """AJAX endpoint to get the encoded dataset HTML"""
    if request.method != "GET":
        return HttpResponse(status=405)
    
    # Check if encoded dataset exists in session
    if "encoded_dataset" not in request.session:
        return HttpResponse(json.dumps({
            "error": "No encoded dataset found. Please enable feature conversion first.",
            "has_encoded_data": False
        }), status=404, content_type="application/json")
    
    try:
        df_encoded = pd.read_json(
            StringIO(request.session["encoded_dataset"]),
            orient="columns"
        )
        
        encoding_info = request.session.get("encoding_info", {})
        
        # Generate full encoded data table for display
        encoded_data_html = df_encoded.to_html(classes="table table-bordered", index=False)
        
        return HttpResponse(json.dumps({
            "success": True,
            "encoding_info": encoding_info,
            "encoded_data_html": encoded_data_html,
            "columns": list(df_encoded.columns),
            "total_rows": len(df_encoded),
            "has_encoded_data": True
        }), content_type="application/json")
        
    except Exception as e:
        return HttpResponse(json.dumps({"error": f"Error retrieving encoded data: {str(e)}"}), status=500, content_type="application/json")


# 
# 4️⃣ FEATURE SELECTION VIEW (Correlation | KBest | RFE | Tree)
# 
def feature_selection_view(request):
    # 
    # DEBUG LOGGING: Show what we're starting with
    # 
    import logging
    logger = logging.getLogger(__name__)
    logger.info("=" * 70)
    logger.info("FEATURE SELECTION REQUEST STARTED")
    logger.info("=" * 70)
    
    # Handle "New Analysis" request - clear all feature-related session data
    if request.method == "POST" and request.POST.get("new_analysis"):
        # Clear ALL feature-related session data for fresh start
        keys_to_delete = []
        for key in request.session.keys():
            # Clear all feature selection related keys
            if any(x in key.lower() for x in ['feature', 'score']):
                if not key.startswith('_auth'):
                    keys_to_delete.append(key)
        
        for key in keys_to_delete:
            try:
                del request.session[key]
            except:
                pass
    
    # Bug C4 fix: use get_dataset() instead of raw pd.read_json() so that
    # file-backed (large) datasets are loaded correctly.
    # NOTE: explicit is None checks — `or` raises ValueError on a non-empty DataFrame.
    df = get_dataset(request, "noise_free_dataset", "noise_free_dataset")
    if df is None:
        df = get_dataset(request, "outlier_free_dataset", "outlier_free_dataset")
    if df is None:
        return redirect("outlier_detection")

    columns = df.columns.tolist()
    logger.info(f"[DEBUG] Dataset columns: {columns}")
    
    # =========================================================================
    # CRITICAL: Get target column EXPLICITLY from request
    # =========================================================================
    # Priority order for target column:
    # 1. POST 'target' parameter (from form)
    # 2. POST 'selected_target' parameter (hidden field)
    # 3. Session 'target_column' (from previous step)
    # 4. GET 'target' parameter
    # 5. Last column as fallback (NOT recommended)
    
    target = None
    
    # First, try to get from POST request
    if request.method == "POST":
        target = request.POST.get("target") or request.POST.get("selected_target")
        logger.info(f"[DEBUG] Target from POST: '{target}'")
    
    # If not in POST, try session
    if not target:
        target = request.session.get("target_column")
        logger.info(f"[DEBUG] Target from session: '{target}'")
    
    # If not in session, try GET
    if not target:
        target = request.GET.get("target")
        logger.info(f"[DEBUG] Target from GET: '{target}'")
    
    # Bug M6 fix: removed silent fallback to last column.
    # If no target is provided, we must show the form and ask the user.
    if not target:
        logger.warning("[WARNING] No target column specified — showing form for user selection.")
        return render(request, "preprocessing/features.html", {
            "selected": False,
            "columns": columns,
            "target": "",
            "error": "Please select a target column before running feature selection.",
        })
    
    logger.info(f"[DEBUG] FINAL UI-TARGET: '{target}'")
    
    # =========================================================================
    # VALIDATION: Ensure target column exists in dataframe
    # =========================================================================
    if target not in df.columns:
        error_msg = (
            f"TARGET COLUMN VALIDATION FAILED!\n"
            f"UI Target: '{target}'\n"
            f"Available columns: {columns}\n"
            f"Please select a valid target column."
        )
        logger.error(error_msg)
        messages.error(request, error_msg)
        return render(request, "preprocessing/features.html", {
            "selected": False,
            "columns": columns,
            "target": target,
        })
    
    # =========================================================================
    # NORMALIZE: Match target column case/format
    # =========================================================================
    target_normalized = target.lower().replace(" ", "_")
    matched_target = None
    for col in df.columns:
        if col.lower().replace(" ", "_") == target_normalized:
            matched_target = col
            break
    
    if matched_target and matched_target != target:
        logger.info(f"[DEBUG] Target normalized: '{target}' → '{matched_target}'")
        target = matched_target
    
    logger.info(f"[DEBUG] VALIDATED BACKEND TARGET: '{target}'")
    
    # Store validated target in session
    request.session["target_column"] = target
    
    # For displaying feature scores
    feature_scores = {}
    selected_features = []

    if request.method == "POST":
        # Check if this is a feature confirmation (not initial method selection)
        selected_raw = request.POST.get("selected_features")
        method = request.POST.get("method")
        
        # =========================================================================
        # CRITICAL: Re-validate target from POST
        # =========================================================================
        target_from_post = request.POST.get("target") or request.POST.get("selected_target")
        if target_from_post:
            # Use the target from POST as the authoritative source
            target_normalized = target_from_post.lower().replace(" ", "_")
            matched_target = None
            for col in df.columns:
                if col.lower().replace(" ", "_") == target_normalized:
                    matched_target = col
                    break
            if matched_target:
                target = matched_target
                logger.info(f"[DEBUG] Target from POST validated: '{target}'")
        
        # Store the validated target
        request.session["target_column"] = target
        logger.info(f"[DEBUG] FINAL FEATURE SELECTION TARGET: '{target}'")
        
        # Normalize target column name to match dataframe columns (handle case/space differences)
        if target:
            target_normalized = target.lower().replace(" ", "_")
            # Try to find a matching column in the dataframe
            for col in df.columns:
                if col.lower().replace(" ", "_") == target_normalized:
                    target = col
                    break
        
        # Parse selected_features (can be JSON string or list from getlist)
        if selected_raw:
            import json as json_module
            try:
                # Try to parse as JSON string
                selected = json_module.loads(selected_raw)
            except (json_module.JSONDecodeError, TypeError):
                # Fall back to getlist
                selected = request.POST.getlist("selected_features")
        else:
            selected = request.POST.getlist("selected_features")
        
        # If selected_features is provided, this is a confirmation - update final_dataset and redirect to ML
        if selected and not method:
            # Bug C4 fix: use get_dataset() for large-dataset support.
            # NOTE: explicit is None checks — `or` raises ValueError on a non-empty DataFrame.
            df = get_dataset(request, "noise_free_dataset", "noise_free_dataset")
            if df is None:
                df = get_dataset(request, "outlier_free_dataset", "outlier_free_dataset")
            if df is None:
                messages.error(request, "Dataset not found. Please start over.")
                return redirect("clean_data")
            
            # Get target from session, with fallback to last column
            target = request.session.get("target_column", "")
            if not target:
                target = df.columns[-1] if len(df.columns) > 0 else ""
            
            # Normalize target column name to match dataframe columns
            if target:
                target_normalized = target.lower().replace(" ", "_")
                matched_target = None
                for col in df.columns:
                    if col.lower().replace(" ", "_") == target_normalized:
                        matched_target = col
                        break
                if matched_target:
                    target = matched_target
            
            # Filter selected features to only include columns that exist in df
            valid_selected = []
            for feat in selected:
                # Try exact match first
                if feat in df.columns:
                    valid_selected.append(feat)
                else:
                    # Try normalized match
                    feat_normalized = feat.lower().replace(" ", "_")
                    for col in df.columns:
                        if col.lower().replace(" ", "_") == feat_normalized:
                            valid_selected.append(col)
                            break
            
            # Ensure target column is included if valid
            if target and target not in valid_selected:
                valid_selected.append(target)
            
            if not valid_selected:
                messages.error(request, "No valid features found in the dataset.")
                return redirect("feature_selection")
            
            # Create final dataset with only selected features
            final_df = df[valid_selected]
            request.session["final_dataset"] = final_df.to_json(orient="columns")
            
            # Store feature selection method for report generation
            request.session["feature_selection_method"] = "manual"
            
            # Save selected features list
            request.session["selected_features"] = valid_selected
            
            # Redirect to Feature Scaling (NEW STEP)
            return redirect("scale_features")
        
        # Validate that a method is selected
        if not method:
            messages.error(request, "Please select a feature selection method.")
            return render(request, "preprocessing/features.html", {
                "selected": False,
                "columns": columns,
                "target": target,
            })
        
        # =====================================================================
        # OPTIONAL: Convert Features to Numeric (Ordinal Encoding)
        # =====================================================================
        # Check if user requested feature encoding
        convert_features = request.POST.get("convert_features") == "true"
        encoding_info = None
        encoded_data_html = None
        
        if convert_features:
            try:
                # Apply ordinal encoding to features (TARGET EXCLUDED)
                # IMPORTANT: convert_features_to_numeric returns (X_encoded, y_unchanged, encoding_info)
                # X_encoded is features-only (target already dropped)
                X_encoded, y_unchanged, encoding_info = convert_features_to_numeric(df, target)
                
                # =========================================================================
                # CRITICAL FIX: Recombine encoded features with target column
                # =========================================================================
                # The target column was temporarily separated for encoding.
                # We MUST reattach it to create the full dataset for feature selection.
                df_encoded = pd.concat([X_encoded, y_unchanged], axis=1)
                
                # =========================================================================
                # MANDATORY VALIDATION: Ensure target is present after encoding
                # =========================================================================
                if target not in df_encoded.columns:
                    error_msg = (
                        f"\n{'=' * 70}\n"
                        f"CRITICAL ERROR: TARGET COLUMN LOST AFTER ENCODING\n"
                        f"{'=' * 70}\n"
                        f"Target column: '{target}'\n"
                        f"Available columns: {list(df_encoded.columns)}\n"
                        f"\nThis indicates a bug in the encoding pipeline.\n"
                        f"The target column should have been reattached after encoding.\n"
                        f"{'=' * 70}\n"
                    )
                    print(error_msg)
                    raise ValueError(
                        f"TARGET COLUMN '{target}' NOT FOUND in encoded dataset.\n"
                        f"Available columns: {list(df_encoded.columns)}\n"
                        f"The target column was lost during feature encoding."
                    )
                
                # =========================================================================
                # MANDATORY LOGGING: Show encoding results and target validation
                # =========================================================================
                print("\n" + "=" * 70)
                print("✅ ENCODING COMPLETE - TARGET VALIDATED")
                print("=" * 70)
                print(f"Target column (PROTECTED): '{target}'")
                print(f"Target present in dataset: {target in df_encoded.columns}")
                print(f"Target dtype: {df_encoded[target].dtype}")
                print(f"Target values preserved: {list(df_encoded[target]) == list(y_unchanged)}")
                print(f"Features encoded: {len(encoding_info['encoded_columns'])} columns")
                print(f"Dataset shape after encoding: {df_encoded.shape}")
                print(f"Original dataset shape: {df.shape}")
                print("=" * 70 + "\n")
                
                # Store encoding info in session for display
                request.session["feature_encoding_applied"] = True
                request.session["encoding_info"] = encoding_info
                
                # Use encoded dataframe (with target reattached) for feature selection
                df = df_encoded
                
                # Generate full encoded data table for display
                encoded_data_html = df.to_html(classes="table table-bordered", index=False)
                
# UI success message - shown ONLY AFTER validation passes
                # Only show message if there were actually columns encoded
                if len(encoding_info['encoded_columns']) > 0:
                    messages.success(
                        request,
                        f"Successfully encoded {len(encoding_info['encoded_columns'])} categorical columns."
                    )
                
            except ValueError as e:
                messages.error(request, str(e))
                return redirect("feature_selection")
        
        # Generate encoding_info if not already set
        if encoding_info is None:
            X_features = df.drop(columns=[target])
            categorical_cols = []
            binary_cols = []
            numeric_cols = []
            for col in X_features.columns:
                if pd.api.types.is_object_dtype(X_features[col]) or pd.api.types.is_categorical_dtype(X_features[col]):
                    categorical_cols.append(col)
                elif pd.api.types.is_numeric_dtype(X_features[col]):
                    nunique = X_features[col].nunique()
                    if nunique <= 2:
                        binary_cols.append(col)
                    else:
                        numeric_cols.append(col)
            encoding_info = {
                'encoded_columns': categorical_cols,
                'numeric_columns': numeric_cols,
                'binary_columns': binary_cols,
                'skipped_columns': []
            }
        
        # Use default K value since form no longer has K input
        k = 5
        
        # Get numeric columns for feature selection
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        print(f"[DEBUG] Numeric columns: {numeric_cols[:5]}...")
        print(f"[DEBUG] Target: '{target}'")
        print(f"[DEBUG] All df columns: {df.columns.tolist()[:10]}...")
        
        # The target has already been validated above, so we can use it directly
        # If target is not in numeric_cols (e.g., categorical target), we still proceed
        # for methods that can handle it (kbest, rfe, tree)
        
        feature_cols = [c for c in numeric_cols if c != target]
        print(f"[DEBUG] Feature cols (after filtering target): {len(feature_cols)} features")
        
        if method == "correlation":
            # Debug: Print what target we received
            print(f"\n[CORRELATION DEBUG] Target from POST: '{target}'")
            print(f"[CORRELATION DEBUG] Columns in df: {df.columns.tolist()[:5]}...")
            
            # First validate that the target column exists and is in the dataframe
            if target not in df.columns:
                messages.error(
                    request,
                    f"Target column '{target}' not found in the dataset. "
                    f"Please select a valid target column."
                )
                return redirect("feature_selection")
            
            print(f"[CORRELATION DEBUG] Target dtype: {df[target].dtype}")
            print(f"[CORRELATION DEBUG] First 3 values: {df[target].head(3).tolist()}")
            
            # Check if target is already numeric using multiple methods
            is_numeric = (
                pd.api.types.is_numeric_dtype(df[target]) or
                pd.api.types.is_integer_dtype(df[target]) or
                pd.api.types.is_float_dtype(df[target])
            )
            
            # Also check if all values can be converted to numeric
            try:
                numeric_values = pd.to_numeric(df[target], errors='raise')
                is_convertible = True
            except:
                is_convertible = False
            
            print(f"[CORRELATION DEBUG] is_numeric: {is_numeric}, is_convertible: {is_convertible}")
            
            # Only proceed if target is already numeric
            if not is_numeric:
                messages.error(
                    request,
                    f"Correlation method requires a numeric target column. "
                    f"Column '{target}' has dtype '{df[target].dtype}' "
                    f"with sample values like: {df[target].iloc[0] if len(df) > 0 else 'N/A'}"
                )
                return redirect("feature_selection")

            corr = correlation_with_target(df, target)
            # Store correlation values for display (range: -1 to +1)
            for feat in corr.index:
                feature_scores[feat] = float(corr[feat])
            
            # =========================================================================
            # CRITICAL FIX: Sort features by score in DESCENDING order (highest first)
            # =========================================================================
            # Sort by absolute correlation (highest absolute value first)
            sorted_features_with_scores = sorted(
                feature_scores.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            sorted_features = [f for f, s in sorted_features_with_scores]
            
            # MANDATORY LOGGING: Show sorted feature order
            print("\n" + "=" * 70)
            print("CORRELATION FEATURE SELECTION - SORTED BY IMPORTANCE (HIGHEST FIRST)")
            print("=" * 70)
            for rank, (feat, score) in enumerate(sorted_features_with_scores, 1):
                print(f"  {rank:2d}. {feat}: {abs(score):.4f} {'(negative)' if score < 0 else ''}")
            print("=" * 70)
            
            selected_features = sorted_features[:k]
            # Target is NOT added to selected features - user selects only features

        elif method == "kbest":
            # Bug H4 fix: choose scorer based on task type.
            # f_classif for classification, f_regression for regression.
            from sklearn.feature_selection import f_classif, f_regression, SelectKBest

            X = df[feature_cols]
            y = df[target]

            # Handle non-numeric targets
            if not pd.api.types.is_numeric_dtype(y):
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y = le.fit_transform(y.astype(str))

            # Detect task type to pick the right scorer
            y_series = pd.Series(y) if isinstance(y, np.ndarray) else y
            is_regression_task = (
                pd.api.types.is_numeric_dtype(y_series)
                and y_series.nunique() > 10
            )
            score_func = f_regression if is_regression_task else f_classif
            scorer_name = "f_regression" if is_regression_task else "f_classif"

            selector = SelectKBest(score_func, k=min(k, len(feature_cols)))
            selector.fit(X, y)

            # Store F-scores
            for i, feat in enumerate(feature_cols):
                feature_scores[feat] = float(selector.scores_[i])

            # Sort features by F-score in DESCENDING order (highest first)
            sorted_features_with_scores = sorted(
                feature_scores.items(),
                key=lambda x: x[1],
                reverse=True,
            )
            sorted_features = [f for f, s in sorted_features_with_scores]

            print("\n" + "=" * 70)
            print(f"K-BEST FEATURE SELECTION - {scorer_name} (HIGHEST FIRST)")
            print("=" * 70)
            for rank, (feat, score) in enumerate(sorted_features_with_scores, 1):
                print(f"  {rank:2d}. {feat}: {score:.4f}")
            print("=" * 70)

            selected_features = sorted_features[:k]
            # Target is NOT added to selected features - user selects only features

        elif method == "rfe":
            # RFE - Recursive Feature Elimination with proper ranking
            from sklearn.feature_selection import RFE
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

            X = df[feature_cols]
            y = df[target]

            # Handle non-numeric targets
            if not pd.api.types.is_numeric_dtype(y):
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y = le.fit_transform(y.astype(str))

            # Convert y to Series if it's an array (for nunique() method)
            if isinstance(y, np.ndarray):
                y_series = pd.Series(y)
            else:
                y_series = y

            # Detect if target is continuous (regression) or discrete (classification)
            # A target is considered continuous if it has many unique values (> 20) and is numeric
            is_regression = (
                pd.api.types.is_numeric_dtype(y_series) and 
                (len(np.unique(y)) > 20 or y_series.dtype in ['float64', 'float32'])
            )
            
            # Use appropriate model based on task type
            if is_regression:
                rf = RandomForestRegressor(n_estimators=100, random_state=42)
            else:
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
            
            rf.fit(X, y)

            # Get importance-based ranking (higher importance = better rank)
            importances = rf.feature_importances_
            sorted_idx = np.argsort(importances)[::-1]  # Descending order

            # =========================================================================
            # CRITICAL FIX: Store IMPORTANCE scores (higher = better) instead of ranks
            # =========================================================================
            # Store importance scores directly (higher importance = better)
            feature_importance_scores = {}
            for i, feat in enumerate(feature_cols):
                feature_importance_scores[feat] = float(importances[i])
                feature_scores[feat] = float(importances[i])  # Use importance, not rank
            
            # Sort by importance (highest first)
            sorted_features_with_scores = sorted(
                feature_importance_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            sorted_features = [f for f, s in sorted_features_with_scores]
            
            # MANDATORY LOGGING: Show sorted feature order
            print("\n" + "=" * 70)
            print("RFE FEATURE SELECTION - SORTED BY IMPORTANCE (HIGHEST FIRST)")
            print("=" * 70)
            for rank, (feat, score) in enumerate(sorted_features_with_scores, 1):
                print(f"  {rank:2d}. {feat}: {score:.4f}")
            print("=" * 70)

            # Select top k features by importance
            selected_features = sorted_features[:k]
            # Target is NOT added to selected features - user selects only features

        elif method == "tree":
            # Tree-based feature importance using RandomForest
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

            X = df[feature_cols]
            y = df[target]

            # Handle non-numeric targets
            if not pd.api.types.is_numeric_dtype(y):
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y = le.fit_transform(y.astype(str))

            # Convert y to Series if it's an array (for nunique() method)
            if isinstance(y, np.ndarray):
                y_series = pd.Series(y)
            else:
                y_series = y

            # Detect if target is continuous (regression) or discrete (classification)
            # A target is considered continuous if it has many unique values (> 20) and is numeric
            is_regression = (
                pd.api.types.is_numeric_dtype(y_series) and 
                (len(np.unique(y)) > 20 or y_series.dtype in ['float64', 'float32'])
            )
            
            # Use appropriate model based on task type
            if is_regression:
                rf = RandomForestRegressor(n_estimators=100, random_state=42)
            else:
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
            
            rf.fit(X, y)

            # Store normalized importance scores (range: 0 to 1, sum = 1)
            importance_sum = rf.feature_importances_.sum()
            feature_importance_scores = {}
            for i, feat in enumerate(feature_cols):
                normalized_importance = float(rf.feature_importances_[i]) / importance_sum if importance_sum > 0 else 0
                normalized_importance = round(normalized_importance, 4)
                feature_importance_scores[feat] = normalized_importance
                feature_scores[feat] = normalized_importance

            # =========================================================================
            # CRITICAL FIX: Sort features by importance in DESCENDING order (highest first)
            # =========================================================================
            sorted_features_with_scores = sorted(
                feature_importance_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            sorted_features = [f for f, s in sorted_features_with_scores]
            
            # MANDATORY LOGGING: Show sorted feature order
            print("\n" + "=" * 70)
            print("TREE-BASED FEATURE SELECTION - SORTED BY IMPORTANCE (HIGHEST FIRST)")
            print("=" * 70)
            for rank, (feat, score) in enumerate(sorted_features_with_scores, 1):
                print(f"  {rank:2d}. {feat}: {score:.4f}")
            print("=" * 70)

            selected_features = sorted_features[:k]
            # Target is NOT added to selected features - user selects only features

        elif method == "none":
            # No feature selection - use all features (excluding target)
            feature_scores = {col: 0.0 for col in feature_cols}
            selected_features = feature_cols
            # Bug M9 fix: sorted_features was not defined for 'none' method,
            # causing a NameError when passed to the template.
            sorted_features = feature_cols

        else:
            messages.error(request, "Invalid feature selection method.")
            return redirect("feature_selection")

        # If user selected specific features from table
        if selected:
            selected_features = selected

        # Ensure target column is included in the final dataset for training/prediction
        if target not in selected_features:
            selected_features.append(target)

        final_df = df[selected_features]

        request.session["final_dataset"] = final_df.to_json(
            orient="columns"
        )
        
        # Store feature selection method and scores for report generation
        request.session["feature_selection_method"] = method
        request.session["feature_scores"] = feature_scores
        
        # Save target column for training
        request.session["target_column"] = target

        # Only pass feature_cols (excluding target) to template for display
        # This ensures the target column is not shown in feature selection tables
        # Note: sorted_features contains features sorted by score (highest first)
        return render(request, "preprocessing/features.html", {
            "selected": True,
            "method": method,
            "features": feature_cols,  # Only input features (X), NOT target (y)
            "features_count": len(feature_cols),  # Count for template display
            "sorted_features_json": json.dumps(sorted_features),  # Pre-sorted by score (highest first)
            "all_features_json": json.dumps(feature_cols),  # All available features as JSON
            "feature_scores_json": json.dumps(feature_scores),
            "preview": final_df.head(10).to_html(classes="table table-bordered"),
            "k": k,
            "target": target,
            "encoding_info": encoding_info,  # Pass encoding info to template for display
            "encoded_data_html": encoded_data_html,  # Full encoded dataset HTML
            "columns": list(df.columns),  # Pass all columns for the Show Data section
        })

    return render(request, "preprocessing/features.html", {
        "selected": False,
        "columns": columns,
        "target": target,
    })

# =========================================================
# 5️⃣ IRRELEVANT FEATURE DETECTION VIEW
# =========================================================
def irrelevant_feature_view(request):
    if "noise_free_dataset" in request.session:
        df = pd.read_json(
            StringIO(request.session["noise_free_dataset"]),
            orient="columns"
        )
    elif "outlier_free_dataset" in request.session:
        df = pd.read_json(
            StringIO(request.session["outlier_free_dataset"]),
            orient="columns"
        )
    elif "cleaned_dataset" in request.session:
        df = pd.read_json(
            StringIO(request.session["cleaned_dataset"]),
            orient="columns"
        )
    else:
        return redirect("clean_data")

    target = request.GET.get("target", df.columns[-1])
    irrelevant = detect_irrelevant_features(df, target)

    return render(request, "preprocessing/irrelevant.html", {
        "irrelevant": irrelevant,
        "count": len(irrelevant),
    })


# =========================================================
# 6️⃣ DOWNLOAD CLEANED DATASET
# =========================================================
def download_cleaned_dataset(request):
    # Bug C6 fix: use get_dataset() so large (file-backed) datasets load correctly.
    df = get_dataset(request, "cleaned_dataset", "cleaned_dataset")
    if df is None:
        return redirect("clean_data")

    response = HttpResponse(content_type="text/csv")
    response["Content-Disposition"] = 'attachment; filename="cleaned_dataset.csv"'
    df.to_csv(response, index=False)
    return response


# =========================================================
# 7️⃣ DOWNLOAD FINAL DATASET
# =========================================================
def download_final_dataset(request):
    # Bug C6 fix: use get_dataset() so large (file-backed) datasets load correctly.
    df = get_dataset(request, "final_dataset", "final_dataset")
    if df is None:
        return redirect("feature_selection")

    response = HttpResponse(content_type="text/csv")
    response["Content-Disposition"] = 'attachment; filename="final_dataset.csv"'
    df.to_csv(response, index=False)
    return response


# =========================================================
# 8️⃣ FEATURE SCALING VIEW - After Feature Selection
# =========================================================
def scale_features_view(request):
    """
    View to apply feature scaling after feature selection.
    
    Pipeline: Feature Selection → Feature Scaling → Model Training
    
    This view:
    1. Loads the selected features from session
    2. Applies scaling ONLY to selected features (not target)
    3. Displays scaled data in a table
    4. Stores scaled data for model training and prediction
    """
    from .feature_scaling import scale_selected_features, get_scaling_method_recommendation
    
    # Check if feature selection has been completed
    if "final_dataset" not in request.session:
        messages.error(request, "Complete feature selection first.")
        return redirect("feature_selection")
    
    if "target_column" not in request.session:
        messages.error(request, "Target column not set.")
        return redirect("feature_selection")
    
    # Bug C5 fix: use get_dataset() instead of raw pd.read_json() so that
    # file-backed (large) datasets are loaded correctly.
    try:
        df = get_dataset(request, "final_dataset", "final_dataset")
        if df is None:
            messages.error(request, "Feature-selected dataset not found. Please redo feature selection.")
            return redirect("feature_selection")
    except Exception as e:
        messages.error(request, "Error loading dataset. Please redo feature selection.")
        logger.error("scale_features_view failed to load final_dataset: %s", e, exc_info=True)
        return redirect("feature_selection")
    
    selected_features = request.session.get("selected_features", list(df.columns))
    target_col = request.session["target_column"]
    
    # Get feature columns only (exclude target)
    feature_cols = [c for c in selected_features if c != target_col]
    
    # Get scaling method from POST or recommend
    method = request.POST.get("scaling_method", "standard")
    
    if request.method == "POST":
        # Apply scaling
        try:
            result = scale_selected_features(
                df=df,
                selected_features=selected_features,
                target_col=target_col,
                method=method
            )
            
            scaled_df = result['scaled_df']
            scaling_info = result['scaling_info']
            
            # =========================================================================
            # CRITICAL: Store scaled data in session for ML pipeline
            # =========================================================================
            # Use file storage for large datasets to preserve ALL rows
            save_dataset(request, scaled_df, "scaled_dataset", "scaled_dataset")
            
            # =========================================================================
            # PRODUCTION-READY SCALER STORAGE
            # =========================================================================
            # Save the COMPLETE fitted scaler object to a file for persistence
            # This is essential for production ML pipelines:
            # - Preserves the exact fit() state learned from training data
            # - Survives server restarts (unlike session storage)
            # - Can be loaded and used directly with scaler.transform()
            # - Supports all scaler types (StandardScaler, MinMaxScaler, RobustScaler)
            scaler_path = save_scaler_to_file(result['scaler'], method, result['feature_cols'])
            
            # Store scaler file path in session for prediction loading
            request.session["scaler_path"] = scaler_path
            
            # Bug C7/M7 fix: scaler_params now stores only the minimal metadata
            # needed to look up the scaler file.  Attribute-based extraction
            # (scaler.mean_, scaler.scale_, etc.) was crashing for MinMaxScaler
            # and RobustScaler.  Prediction now loads the full scaler object
            # from file (load_scaler_for_prediction) and calls scaler.transform().
            scaler_params = {
                'method': method,
                'feature_cols': result['feature_cols'],
                'target_col': target_col,
                'feature_order': result['feature_cols'],
                # Store mean/scale only when the scaler has these attrs (StandardScaler)
                'scaler_mean': (
                    {col: float(result['scaler'].mean_[i])
                     for i, col in enumerate(result['feature_cols'])
                     if col in scaling_info['numeric_features_scaled']}
                    if hasattr(result['scaler'], 'mean_') else {}
                ),
                'scaler_scale': (
                    {col: float(result['scaler'].scale_[i])
                     for i, col in enumerate(result['feature_cols'])
                     if col in scaling_info['numeric_features_scaled']}
                    if hasattr(result['scaler'], 'scale_') else {}
                ),
            }
            request.session["scaler_params"] = scaler_params
            
            # =========================================================================
            # Prepare scaled data for template display (ALL rows)
            # =========================================================================
            scaled_rows = []
            feature_columns = result['scaling_info']['feature_columns']
            
            for idx, row in scaled_df.iterrows():
                row_data = {
                    'index': idx + 1,
                    'values': {},
                    'target': row[target_col]
                }
                # Add each scaled feature value
                for col in feature_columns:
                    val = row[col]
                    if pd.notna(val):
                        if isinstance(val, (np.floating, float)):
                            row_data['values'][col] = round(float(val), 4)
                        else:
                            row_data['values'][col] = val
                    else:
                        row_data['values'][col] = '-'
                scaled_rows.append(row_data)
            
            return render(request, "preprocessing/scale_features.html", {
                'scaled_rows': scaled_rows,
                'scaling_info': result['scaling_info'],
                'method': method,
                'scaled_data_json': result['scaled_data_json'],
            })
            
        except Exception as e:
            messages.error(request, f"Error scaling features: {str(e)}")
            return redirect("scale_features")
    
    # GET request - show scaling options
    recommended_method = get_scaling_method_recommendation(df, selected_features, target_col)
    
    # Get numeric columns for scaling info
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    return render(request, "preprocessing/scale_features.html", {
        'selected_features': feature_cols,
        'target_col': target_col,
        'recommended_method': recommended_method,
        'n_samples': len(df),
        'n_features': len(feature_cols),
        'n_numeric_features': len(numeric_cols),
        'numeric_features': numeric_cols,
    })


def model_training_view(request):
    # Prevent redirect loops when user arrives directly from a non-linear entry
    # point (e.g., direct Visualization access). Only enforce feature selection
    # if we are not already in the middle of a visualization-started flow.
    if "final_dataset" not in request.session:
        if request.session.get("workflow_stage") != "visualization":
            return redirect("feature_selection")

    return render(request, "preprocessing/model_training.html")

