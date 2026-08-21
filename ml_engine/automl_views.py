"""
ml_engine/automl_views.py
==========================
AutoML Workflow Views for InsightForge.

Provides a one-click AutoML pipeline that sequentially:
    1. Analyzes dataset
    2. Applies intelligent imputation (auto-recommended)
    3. Applies intelligent encoding (auto-recommended)
    4. Runs feature selection (correlation method, top 10)
    5. Applies feature scaling (auto-recommended)
    6. Recommends and trains top 3 models
    7. Selects best model based on primary metric
    8. Saves pipeline + model
    9. Sets up session for prediction and report generation

Progress is tracked in the Django session and polled via AJAX.

MANUAL MODE is always accessible — AutoML never removes the existing workflow.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict

import joblib
import numpy as np
import pandas as pd

from django.conf import settings
from django.contrib import messages
from django.http import JsonResponse
from django.shortcuts import render, redirect

from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    RandomForestRegressor, GradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, f1_score, r2_score,
    mean_absolute_error, mean_squared_error,
)
from sklearn.model_selection import train_test_split

from preprocessing.imputation import analyze_dataframe_for_imputation, apply_imputation
from preprocessing.encoding import analyze_dataframe_for_encoding, EncodingApplier
from preprocessing.views import get_dataset, save_dataset
from .recommender import analyze_dataset, generate_recommendation_summary

logger = logging.getLogger(__name__)


# ============================================================================
# PROGRESS STAGES
# ============================================================================

AUTOML_STAGES = [
    {'key': 'analyzing',   'label': 'Analyzing Dataset',    'icon': '🔍'},
    {'key': 'imputing',    'label': 'Handling Missing Values', 'icon': '🧹'},
    {'key': 'encoding',    'label': 'Encoding Features',    'icon': '🔢'},
    {'key': 'selecting',   'label': 'Selecting Features',   'icon': '🎯'},
    {'key': 'scaling',     'label': 'Scaling Features',     'icon': '⚖️'},
    {'key': 'training',    'label': 'Training Models',      'icon': '🤖'},
    {'key': 'evaluating',  'label': 'Evaluating & Selecting', 'icon': '📊'},
    {'key': 'saving',      'label': 'Saving Pipeline',      'icon': '💾'},
    {'key': 'complete',    'label': 'AutoML Complete!',     'icon': '✅'},
]


def _set_progress(request, stage_key: str, message: str = '', progress_pct: int = 0):
    """Update AutoML progress in session."""
    request.session['automl_stage'] = stage_key
    request.session['automl_message'] = message
    request.session['automl_progress'] = progress_pct
    request.session['automl_stages'] = AUTOML_STAGES
    request.session.modified = True


# ============================================================================
# AUTOML STATUS ENDPOINT (AJAX polling)
# ============================================================================

def automl_status_view(request):
    """
    AJAX endpoint that returns the current AutoML progress.
    Called by the frontend every 1 second during AutoML execution.
    """
    return JsonResponse({
        'stage': request.session.get('automl_stage', 'idle'),
        'message': request.session.get('automl_message', ''),
        'progress': request.session.get('automl_progress', 0),
        'stages': AUTOML_STAGES,
        'is_complete': request.session.get('automl_stage') == 'complete',
        'has_error': request.session.get('automl_stage') == 'error',
        'error_message': request.session.get('automl_error', ''),
    })


# ============================================================================
# AUTOML MAIN VIEW
# ============================================================================

def automl_view(request):
    """
    AutoML workflow view.

    GET:  Show AutoML setup form (target column selection, options)
    POST: Execute full AutoML pipeline and render results
    """
    # Load dataset
    df = None
    for key in ('dataset', 'cleaned_dataset', 'outlier_free_dataset', 'noise_free_dataset'):
        df = get_dataset(request, key, key)
        if df is not None:
            break

    if df is None:
        messages.error(request, "Please upload a dataset first to use AutoML.")
        return redirect('upload_dataset')

    columns = df.columns.tolist()

    if request.method == 'GET':
        return render(request, 'ml_engine/automl.html', {
            'columns': columns,
            'stages': AUTOML_STAGES,
            'n_rows': len(df),
            'n_cols': len(df.columns),
        })

    # POST — Run AutoML
    target_col = request.POST.get('target', '')
    if not target_col or target_col not in df.columns:
        messages.error(request, f"Invalid target column: '{target_col}'")
        return render(request, 'ml_engine/automl.html', {
            'columns': columns,
            'stages': AUTOML_STAGES,
            'error': f"Invalid target column selected.",
        })

    try:
        results = _run_automl_pipeline(request, df, target_col)
        return render(request, 'ml_engine/automl.html', {
            'columns': columns,
            'stages': AUTOML_STAGES,
            'results': results,
            'completed': True,
        })

    except Exception as e:
        logger.exception(f"AutoML pipeline failed: {e}")
        request.session['automl_stage'] = 'error'
        request.session['automl_error'] = str(e)
        return render(request, 'ml_engine/automl.html', {
            'columns': columns,
            'stages': AUTOML_STAGES,
            'error': str(e),
        })


# ============================================================================
# AUTOML PIPELINE EXECUTION
# ============================================================================

def _run_automl_pipeline(request, df: pd.DataFrame, target_col: str) -> Dict:
    """
    Execute the complete AutoML pipeline sequentially.

    Returns a results dict for template rendering.
    """
    results = {
        'target': target_col,
        'stages_completed': [],
    }

    # ── Stage 1: Analyze Dataset ─────────────────────────────────────────────
    _set_progress(request, 'analyzing', f'Profiling {len(df):,} rows × {len(df.columns)} features...', 5)
    profile = analyze_dataset(df, target_col)
    task_type = profile['task_type']
    results['profile'] = profile
    results['task_type'] = task_type
    results['stages_completed'].append('analyzing')
    logger.info(f"AutoML: Dataset profiled — {task_type}, {profile['n_rows']} rows, {profile['n_features']} features")

    # ── Stage 2: Imputation ───────────────────────────────────────────────────
    _set_progress(request, 'imputing', 'Applying intelligent imputation...', 15)
    imputation_analyses = analyze_dataframe_for_imputation(df)
    column_impute_config = {a['column']: a['recommended'] for a in imputation_analyses}
    df_imputed = apply_imputation(df, column_config=column_impute_config)
    results['imputation_report'] = {
        'columns_imputed': len(imputation_analyses),
        'methods_used': {a['column']: a['recommended'] for a in imputation_analyses},
    }
    results['stages_completed'].append('imputing')

    # ── Stage 3: Encoding ─────────────────────────────────────────────────────
    _set_progress(request, 'encoding', 'Encoding categorical features...', 30)
    encoding_analyses = analyze_dataframe_for_encoding(df_imputed, target_col)
    column_encode_config = {
        a['column']: a['recommended']
        for a in encoding_analyses
        if a['needs_encoding'] and not a['is_target']
    }
    applier = EncodingApplier()
    df_encoded = applier.fit_transform(df_imputed, target_col, column_encode_config)
    results['encoding_report'] = {
        'columns_encoded': len(column_encode_config),
        'methods_used': column_encode_config,
    }
    results['stages_completed'].append('encoding')

    # ── Stage 4: Feature Selection ────────────────────────────────────────────
    _set_progress(request, 'selecting', 'Selecting most relevant features...', 45)
    y = df_encoded[target_col]
    X = df_encoded.drop(columns=[target_col])
    X_numeric = X.select_dtypes(include=[np.number])

    selected_features = list(X_numeric.columns)  # Default: all
    if len(X_numeric.columns) > 1 and len(X_numeric.columns) <= 50:
        try:
            corr = X_numeric.corrwith(y.astype(float) if pd.api.types.is_numeric_dtype(y) else y.cat.codes).abs()
            k = min(10, len(X_numeric.columns))
            selected_features = corr.nlargest(k).index.tolist()
        except Exception as e:
            logger.warning(f"AutoML feature selection failed: {e}. Using all features.")
            selected_features = list(X_numeric.columns)

    df_selected = df_encoded[selected_features + [target_col]]
    results['selection_report'] = {
        'n_selected': len(selected_features),
        'selected_features': selected_features,
    }
    results['stages_completed'].append('selecting')

    # Save feature selection to session
    request.session['selected_features'] = selected_features
    request.session['target_column'] = target_col

    # ── Stage 5: Scaling ──────────────────────────────────────────────────────
    _set_progress(request, 'scaling', 'Scaling features...', 55)
    from preprocessing.feature_scaling import get_scaler, get_scaling_method_recommendation
    recommended_scaler = get_scaling_method_recommendation(df_selected, selected_features, target_col)

    X_sel = df_selected.drop(columns=[target_col])
    numeric_feat_cols = X_sel.select_dtypes(include=[np.number]).columns.tolist()
    scaler = get_scaler(recommended_scaler)
    X_scaled = X_sel.copy()
    if numeric_feat_cols:
        X_scaled[numeric_feat_cols] = scaler.fit_transform(X_sel[numeric_feat_cols])

    df_scaled = X_scaled.copy()
    df_scaled[target_col] = df_selected[target_col].values
    results['scaling_report'] = {
        'method': recommended_scaler,
        'features_scaled': len(numeric_feat_cols),
    }
    results['stages_completed'].append('scaling')

    # ── Stage 6: Train Top Recommended Models ─────────────────────────────────
    _set_progress(request, 'training', 'Training recommended models...', 65)
    recommendation_summary = generate_recommendation_summary(profile)
    top_recs = recommendation_summary['top_recommendations']
    results['recommendations'] = top_recs

    X_final = df_scaled.drop(columns=[target_col])
    y_final = df_scaled[target_col]

    # Ensure X is all numeric
    X_final = X_final.select_dtypes(include=[np.number])
    if X_final.empty:
        raise ValueError("No numeric features available for training after preprocessing.")

    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y_final, test_size=0.25, random_state=42
    )

    # Encode target for classification if needed
    if task_type == 'classification' and not pd.api.types.is_numeric_dtype(y_train):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_train = pd.Series(le.fit_transform(y_train.astype(str)), name=target_col)
        y_test = pd.Series(le.transform(y_test.astype(str)), name=target_col)
        request.session['automl_label_encoder'] = list(le.classes_)

    # Build model map
    MODEL_MAP_CLF = {
        'random_forest': RandomForestClassifier(n_estimators=200, random_state=42),
        'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
        'decision_tree': None,   # Not in top3 usually, but handled by fallback
        'svm': None,             # Slow — skipped in AutoML
        'naive_bayes': None,
    }
    MODEL_MAP_REG = {
        'random_forest_regressor': RandomForestRegressor(n_estimators=200, random_state=42),
        'gradient_boosting_regressor': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'linear_regression': LinearRegression(),
        'svr': None,             # Slow — skipped in AutoML
    }
    model_map = MODEL_MAP_CLF if task_type == 'classification' else MODEL_MAP_REG

    trained_models = {}
    for rec in top_recs:
        key = rec['model_key']
        model = model_map.get(key)
        if model is None:
            continue
        try:
            model.fit(X_train, y_train)
            trained_models[rec['model_name']] = {
                'model': model,
                'key': key,
                'recommendation': rec,
            }
        except Exception as e:
            logger.warning(f"AutoML: Failed to train {rec['model_name']}: {e}")

    if not trained_models:
        raise RuntimeError("No models could be trained. Check the dataset and target column.")

    results['stages_completed'].append('training')

    # ── Stage 7: Evaluate & Select Best ───────────────────────────────────────
    _set_progress(request, 'evaluating', 'Evaluating models and selecting best...', 80)
    model_results = []
    best_model_name = None
    best_model_obj = None
    best_score = -np.inf

    for name, info in trained_models.items():
        model = info['model']
        try:
            y_pred = model.predict(X_test)
            if task_type == 'classification':
                acc = round(accuracy_score(y_test, y_pred) * 100, 2)
                f1 = round(f1_score(y_test, y_pred, average='weighted', zero_division=0) * 100, 2)
                score = f1
                model_results.append({
                    'name': name,
                    'accuracy': acc,
                    'f1': f1,
                    'score': score,
                    'recommendation': info['recommendation'],
                })
            else:
                r2 = round(r2_score(y_test, y_pred) * 100, 2)
                mae = round(mean_absolute_error(y_test, y_pred), 4)
                rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)), 4)
                score = r2
                model_results.append({
                    'name': name,
                    'r2': r2,
                    'mae': mae,
                    'rmse': rmse,
                    'score': score,
                    'recommendation': info['recommendation'],
                })

            if score > best_score:
                best_score = score
                best_model_name = name
                best_model_obj = model

        except Exception as e:
            logger.warning(f"AutoML: Evaluation failed for {name}: {e}")

    results['model_results'] = model_results
    results['best_model'] = best_model_name
    results['best_score'] = best_score
    results['stages_completed'].append('evaluating')

    # ── Stage 8: Save Pipeline + Model ────────────────────────────────────────
    _set_progress(request, 'saving', 'Saving pipeline and best model...', 90)
    model_dir = os.path.join(settings.BASE_DIR, 'saved_models')
    os.makedirs(model_dir, exist_ok=True)

    # Save model
    session_key = request.session.session_key or 'automl'
    model_path = os.path.join(model_dir, f'automl_model_{session_key}.pkl')
    joblib.dump(best_model_obj, model_path)

    # Save scaler
    scaler_path = os.path.join(model_dir, f'automl_scaler_{session_key}.pkl')
    joblib.dump(scaler, scaler_path)

    # Save encoded dataset for report
    save_dataset(request, df_scaled, 'scaled_dataset', 'scaled_dataset')
    save_dataset(request, df_encoded, 'encoded_dataset', 'encoded_dataset')
    save_dataset(request, df_imputed, 'cleaned_dataset', 'cleaned_dataset')

    # Store in session for downstream steps
    request.session['model_path'] = model_path
    request.session['scaler_path'] = scaler_path
    request.session['model_features'] = selected_features
    request.session['model_target'] = target_col
    request.session['model_is_classification'] = (task_type == 'classification')
    request.session['task_type'] = task_type
    request.session['automl_completed'] = True

    # Session report data
    best_result = next((m for m in model_results if m['name'] == best_model_name), {})
    if task_type == 'classification':
        request.session['model_report'] = {
            'best_model': best_model_name,
            'accuracy': best_result.get('accuracy', 0),
            'f1': best_result.get('f1', 0),
            'task_type': 'classification',
            'automl': True,
        }
    else:
        request.session['model_report'] = {
            'best_model': best_model_name,
            'r2': best_result.get('r2', 0),
            'mae': best_result.get('mae', 0),
            'rmse': best_result.get('rmse', 0),
            'task_type': 'regression',
            'automl': True,
        }

    results['stages_completed'].append('saving')

    # ── Stage 9: Complete ─────────────────────────────────────────────────────
    _set_progress(request, 'complete', 'AutoML pipeline completed successfully!', 100)
    results['stages_completed'].append('complete')
    results['scaler_method'] = recommended_scaler
    results['feature_columns'] = selected_features
    results['task_type'] = task_type

    return results
