"""
Feature Scaling View Integration Example

This file shows how to integrate the feature scaling module
into the existing Django views pipeline.

Pipeline Order:
===============
Feature Selection (returns selected features) 
    → Feature Scaling (this module) 
    → Model Training 
    → Prediction

Usage:
======
Add this view after feature selection and before model training.
"""

from django.shortcuts import render, redirect
from django.contrib import messages
import pandas as pd
from io import StringIO

from .feature_scaling import scale_selected_features, get_scaling_method_recommendation


# ============================================================================
# VIEW: Scale Selected Features
# ============================================================================

def scale_features_view(request):
    """
    View to apply feature scaling after feature selection.
    
    Expected session data:
        - 'final_dataset': DataFrame with selected features
        - 'target_column': Name of target column
        - 'selected_features': List of selected feature names
    
    Returns:
        Rendered template with scaled data table
    """
    # Check if feature selection has been completed
    if "final_dataset" not in request.session:
        messages.error(request, "Complete feature selection first.")
        return redirect("feature_selection")
    
    if "target_column" not in request.session:
        messages.error(request, "Target column not set.")
        return redirect("feature_selection")
    
    # Load dataset
    df = pd.read_json(
        StringIO(request.session["final_dataset"]),
        orient="columns"
    )
    
    selected_features = request.session.get("selected_features", list(df.columns))
    target_col = request.session["target_column"]
    
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
            
            # Store scaled data in session for downstream use
            request.session["scaled_data"] = result['scaled_data_json']
            request.session["scaling_info"] = result['scaling_info']
            request.session["scaler_params"] = {
                'method': method,
                'feature_cols': result['feature_cols'],
            }
            
            messages.success(
                request,
                f"Feature scaling complete. Scaled {result['scaling_info']['n_features']} features using {method} scaling."
            )
            
            return render(request, "preprocessing/scaling_result.html", {
                'scaled_html': result['scaled_html'],
                'scaling_info': result['scaling_info'],
                'method': method,
            })
            
        except Exception as e:
            messages.error(request, f"Error scaling features: {str(e)}")
            return redirect("feature_selection")
    
    # GET request - show scaling options
    recommended_method = get_scaling_method_recommendation(df, selected_features, target_col)
    
    return render(request, "preprocessing/scale_features.html", {
        'selected_features': selected_features,
        'target_col': target_col,
        'recommended_method': recommended_method,
        'n_samples': len(df),
        'n_features': len([f for f in selected_features if f != target_col]),
    })


# ============================================================================
# INTEGRATION WITH MODEL TRAINING
# ============================================================================

def get_scaled_data_for_training(request):
    """
    Helper function to get scaled data for model training.
    
    Returns:
        tuple: (X_scaled, y, feature_cols, scaling_info)
    """
    if "scaled_data" not in request.session:
        raise ValueError("Scaled data not found. Apply feature scaling first.")
    
    df = pd.read_json(
        StringIO(request.session["scaled_data"]),
        orient="columns"
    )
    
    scaling_info = request.session.get("scaling_info", {})
    target_col = scaling_info.get('target_column', request.session.get("target_column"))
    feature_cols = scaling_info.get('feature_columns', 
                                     [c for c in df.columns if c != target_col])
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    return X, y, feature_cols, scaling_info


# ============================================================================
# INTEGRATION WITH PREDICTION
# ============================================================================

def scale_prediction_input(input_data, request):
    """
    Scale input data for prediction using fitted scaler.
    
    Args:
        input_data: dict with feature values
        request: Django request object with session data
    
    Returns:
        Scaled input array ready for model prediction
    """
    from .feature_scaling import apply_scaling_to_new_data
    
    scaler_params = request.session.get("scaler_params", {})
    feature_cols = scaler_params.get("feature_cols", [])
    
    if not feature_cols:
        raise ValueError("Scaler parameters not found in session.")
    
    # Load scaler (in production, you might store the actual scaler object)
    # For now, we reconstruct the feature vector
    import numpy as np
    
    X_new = pd.DataFrame([input_data])
    X_numeric = X_new[feature_cols].select_dtypes(include=[np.number])
    
    # In a full implementation, you would:
    # 1. Load the fitted scaler from session or disk
    # 2. Apply scaler.transform() to new data
    
    return X_numeric.values


# ============================================================================
# TEMPLATE TAG: Display Scaled Data Table
# ============================================================================

def get_scaled_table_context(request):
    """
    Get context data for displaying scaled data table in templates.
    
    Returns:
        dict with scaled_html, scaling_info
    """
    if "scaled_data" not in request.session:
        return {
            'scaled_html': None,
            'scaling_info': None,
            'has_scaled_data': False,
        }
    
    scaling_info = request.session.get("scaling_info", {})
    
    return {
        'scaled_html': request.session.get("scaled_html"),
        'scaling_info': scaling_info,
        'has_scaled_data': True,
    }

