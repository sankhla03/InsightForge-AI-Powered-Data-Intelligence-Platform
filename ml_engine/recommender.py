"""
ml_engine/recommender.py
=========================
Intelligent Model Recommendation Engine for InsightForge.

Analyzes the dataset profile and recommends the most suitable ML algorithms
BEFORE training. Users can accept, filter, or override recommendations.

PROFILE FACTORS ANALYZED:
    - Dataset size (n_rows, n_features)
    - Task type (classification vs regression)
    - Missing value ratio
    - Feature types (numeric vs categorical)
    - Class imbalance (classification only)
    - Dataset dimensionality

RECOMMENDATION OUTPUT (per model):
    - model_name        : Algorithm name
    - reason            : Why this model suits the dataset
    - expected_accuracy : Qualitative estimate (High / Medium / Low)
    - training_speed    : Qualitative estimate (Fast / Medium / Slow)
    - interpretability  : Qualitative estimate (High / Medium / Low)
    - memory_usage      : Qualitative estimate (Low / Medium / High)
    - rank              : Priority rank (1 = top recommendation)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# DATASET PROFILER
# ============================================================================

def analyze_dataset(df: pd.DataFrame, target_col: str) -> Dict:
    """
    Build a comprehensive profile of the dataset for model recommendation.

    Args:
        df: Input DataFrame (after preprocessing)
        target_col: Name of the target column

    Returns:
        Profile dict with keys:
            n_rows, n_features, missing_pct, task_type, n_classes,
            has_class_imbalance, imbalance_ratio, is_large_dataset,
            is_small_dataset, is_high_dimensional, numeric_feature_ratio,
            has_mixed_types, feature_names
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    y = df[target_col]
    X = df.drop(columns=[target_col])

    n_rows = len(df)
    n_features = len(X.columns)

    # Missing value ratio
    total_cells = n_rows * n_features
    missing_cells = X.isnull().sum().sum()
    missing_pct = round((missing_cells / total_cells) * 100, 2) if total_cells > 0 else 0.0

    # Task type detection
    task_type = _detect_task_type(y)

    # Class statistics (classification only)
    n_classes = None
    has_class_imbalance = False
    imbalance_ratio = 1.0

    if task_type == 'classification':
        n_classes = y.nunique()
        value_counts = y.value_counts()
        if len(value_counts) >= 2:
            majority = value_counts.iloc[0]
            minority = value_counts.iloc[-1]
            imbalance_ratio = round(majority / minority, 2)
            has_class_imbalance = imbalance_ratio > 3.0  # 3:1 threshold

    # Feature type analysis
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    numeric_feature_ratio = len(numeric_cols) / max(n_features, 1)
    has_mixed_types = len(cat_cols) > 0 and len(numeric_cols) > 0

    # Size categories
    is_large_dataset = n_rows > 50_000
    is_very_large_dataset = n_rows > 500_000
    is_small_dataset = n_rows < 500
    is_high_dimensional = n_features > 50

    return {
        'n_rows': n_rows,
        'n_features': n_features,
        'missing_pct': missing_pct,
        'task_type': task_type,
        'n_classes': n_classes,
        'has_class_imbalance': has_class_imbalance,
        'imbalance_ratio': imbalance_ratio,
        'is_large_dataset': is_large_dataset,
        'is_very_large_dataset': is_very_large_dataset,
        'is_small_dataset': is_small_dataset,
        'is_high_dimensional': is_high_dimensional,
        'numeric_feature_ratio': round(numeric_feature_ratio, 2),
        'has_mixed_types': has_mixed_types,
        'n_numeric_features': len(numeric_cols),
        'n_categorical_features': len(cat_cols),
        'feature_names': list(X.columns),
    }


def _detect_task_type(y: pd.Series, threshold: int = 10) -> str:
    """Detect classification vs regression based on target column properties."""
    if pd.api.types.is_float_dtype(y):
        return 'regression'
    if pd.api.types.is_numeric_dtype(y) and y.nunique() >= threshold:
        return 'regression'
    return 'classification'


# ============================================================================
# MODEL RECOMMENDATION ENGINE
# ============================================================================

# Classification model catalog
CLASSIFICATION_MODELS = [
    {
        'key': 'random_forest',
        'name': 'Random Forest Classifier',
        'base_reason': 'Handles mixed features, robust to outliers, minimal tuning needed.',
        'expected_accuracy': 'High',
        'training_speed': 'Medium',
        'interpretability': 'Medium',
        'memory_usage': 'Medium',
        'base_score': 85,
        'large_dataset_penalty': 15,  # Slower on very large datasets
        'small_dataset_bonus': 10,    # Works well on small datasets too
        'imbalance_handling': 'Built-in class_weight support',
    },
    {
        'key': 'gradient_boosting',
        'name': 'Gradient Boosting Classifier',
        'base_reason': 'Best for structured tabular data. Sequential ensemble that corrects errors iteratively.',
        'expected_accuracy': 'High',
        'training_speed': 'Slow',
        'interpretability': 'Low',
        'memory_usage': 'Medium',
        'base_score': 88,
        'large_dataset_penalty': 25,
        'small_dataset_bonus': 5,
        'imbalance_handling': 'Moderate sensitivity to imbalance',
    },
    {
        'key': 'logistic_regression',
        'name': 'Logistic Regression',
        'base_reason': 'Excellent interpretable baseline. Fast training, works best with linear decision boundaries.',
        'expected_accuracy': 'Medium',
        'training_speed': 'Fast',
        'interpretability': 'High',
        'memory_usage': 'Low',
        'base_score': 72,
        'large_dataset_penalty': 0,
        'small_dataset_bonus': 15,
        'imbalance_handling': 'Built-in class_weight support',
    },
    {
        'key': 'decision_tree',
        'name': 'Decision Tree Classifier',
        'base_reason': 'Highly interpretable. Good for understanding decision boundaries.',
        'expected_accuracy': 'Medium',
        'training_speed': 'Fast',
        'interpretability': 'High',
        'memory_usage': 'Low',
        'base_score': 65,
        'large_dataset_penalty': 5,
        'small_dataset_bonus': 8,
        'imbalance_handling': 'Sensitive to imbalance',
    },
    {
        'key': 'svm',
        'name': 'Support Vector Machine (SVM)',
        'base_reason': 'Effective in high-dimensional spaces. Works well when classes are clearly separable.',
        'expected_accuracy': 'High',
        'training_speed': 'Slow',
        'interpretability': 'Low',
        'memory_usage': 'High',
        'base_score': 78,
        'large_dataset_penalty': 30,
        'small_dataset_bonus': 20,
        'imbalance_handling': 'Built-in class_weight support',
    },
    {
        'key': 'naive_bayes',
        'name': 'Naive Bayes',
        'base_reason': 'Very fast probabilistic classifier. Good baseline for text or independent features.',
        'expected_accuracy': 'Low',
        'training_speed': 'Fast',
        'interpretability': 'High',
        'memory_usage': 'Low',
        'base_score': 60,
        'large_dataset_penalty': 0,
        'small_dataset_bonus': 5,
        'imbalance_handling': 'Sensitive to imbalance',
    },
]

# Regression model catalog
REGRESSION_MODELS = [
    {
        'key': 'random_forest_regressor',
        'name': 'Random Forest Regressor',
        'base_reason': 'Robust ensemble method that handles non-linear relationships and mixed features well.',
        'expected_accuracy': 'High',
        'training_speed': 'Medium',
        'interpretability': 'Medium',
        'memory_usage': 'Medium',
        'base_score': 85,
        'large_dataset_penalty': 15,
        'small_dataset_bonus': 10,
    },
    {
        'key': 'gradient_boosting_regressor',
        'name': 'Gradient Boosting Regressor',
        'base_reason': 'State-of-the-art for structured regression tasks. Handles complex patterns.',
        'expected_accuracy': 'High',
        'training_speed': 'Slow',
        'interpretability': 'Low',
        'memory_usage': 'Medium',
        'base_score': 88,
        'large_dataset_penalty': 25,
        'small_dataset_bonus': 5,
    },
    {
        'key': 'linear_regression',
        'name': 'Linear Regression',
        'base_reason': 'Fast interpretable baseline. Ideal when the relationship is approximately linear.',
        'expected_accuracy': 'Medium',
        'training_speed': 'Fast',
        'interpretability': 'High',
        'memory_usage': 'Low',
        'base_score': 68,
        'large_dataset_penalty': 0,
        'small_dataset_bonus': 15,
    },
    {
        'key': 'svr',
        'name': 'Support Vector Regressor (SVR)',
        'base_reason': 'Effective for non-linear regression with a small number of features.',
        'expected_accuracy': 'High',
        'training_speed': 'Slow',
        'interpretability': 'Low',
        'memory_usage': 'High',
        'base_score': 75,
        'large_dataset_penalty': 30,
        'small_dataset_bonus': 20,
    },
]


def recommend_models(profile: Dict) -> List[Dict]:
    """
    Recommend ML models based on dataset profile.

    Args:
        profile: Output from analyze_dataset()

    Returns:
        Ranked list of model recommendation dicts, each with:
            rank, model_key, model_name, score, reason,
            expected_accuracy, training_speed, interpretability,
            memory_usage, warnings
    """
    task_type = profile['task_type']
    catalog = CLASSIFICATION_MODELS if task_type == 'classification' else REGRESSION_MODELS

    recommendations = []

    for model_info in catalog:
        score = model_info['base_score']
        reasons = [model_info['base_reason']]
        warnings = []

        # === ADJUSTMENTS BASED ON PROFILE ===

        # Large dataset penalty
        if profile['is_very_large_dataset']:
            score -= model_info['large_dataset_penalty']
            if model_info['large_dataset_penalty'] > 20:
                warnings.append(f"⚠️ Very large dataset ({profile['n_rows']:,} rows) — training may be slow.")

        elif profile['is_large_dataset']:
            score -= model_info['large_dataset_penalty'] // 2

        # Small dataset bonus
        if profile['is_small_dataset']:
            score += model_info['small_dataset_bonus']
            if model_info['key'] in ('svm', 'svr'):
                reasons.append(f"SVM performs particularly well on small, well-structured datasets.")
            if profile['n_rows'] < 100:
                warnings.append(f"⚠️ Very small dataset ({profile['n_rows']} rows) — risk of overfitting.")

        # High dimensionality
        if profile['is_high_dimensional']:
            if model_info['key'] in ('random_forest', 'gradient_boosting',
                                      'random_forest_regressor', 'gradient_boosting_regressor'):
                score += 5
                reasons.append(f"Tree-based models handle high dimensionality ({profile['n_features']} features) well with built-in feature selection.")
            elif model_info['key'] in ('svm', 'svr'):
                score += 8
                reasons.append("SVM is effective in high-dimensional feature spaces.")
            elif model_info['key'] == 'logistic_regression':
                score += 3
                reasons.append("Logistic Regression with L2 regularization handles high dimensions.")

        # Mixed feature types bonus for tree-based models
        if profile['has_mixed_types'] and model_info['key'] in (
            'random_forest', 'gradient_boosting', 'decision_tree',
            'random_forest_regressor', 'gradient_boosting_regressor',
        ):
            score += 5
            reasons.append(f"Handles mixed feature types ({profile['n_numeric_features']} numeric + {profile['n_categorical_features']} categorical).")

        # Class imbalance warnings (classification)
        if task_type == 'classification' and profile['has_class_imbalance']:
            imbalance_ratio = profile['imbalance_ratio']
            warnings.append(
                f"⚠️ Class imbalance detected (ratio {imbalance_ratio}:1). "
                f"Consider using class_weight='balanced' or SMOTE."
            )
            if model_info['key'] in ('random_forest', 'logistic_regression', 'svm'):
                score += 5  # These have built-in class_weight support

        # Missing data handling
        if profile['missing_pct'] > 10:
            if model_info['key'] in ('random_forest', 'gradient_boosting',
                                      'random_forest_regressor', 'gradient_boosting_regressor'):
                reasons.append(f"Tree models are robust to remaining missing patterns after imputation.")

        # Build final reason string
        final_reason = reasons[0]
        if len(reasons) > 1:
            final_reason += " " + " ".join(reasons[1:])

        recommendations.append({
            'rank': 0,  # Will be set after sorting
            'model_key': model_info['key'],
            'model_name': model_info['name'],
            'score': score,
            'reason': final_reason,
            'expected_accuracy': model_info['expected_accuracy'],
            'training_speed': model_info['training_speed'],
            'interpretability': model_info['interpretability'],
            'memory_usage': model_info['memory_usage'],
            'warnings': warnings,
        })

    # Sort by score descending
    recommendations.sort(key=lambda x: x['score'], reverse=True)

    # Assign ranks
    for i, rec in enumerate(recommendations):
        rec['rank'] = i + 1

    return recommendations


def get_top_n_recommendations(profile: Dict, n: int = 3) -> List[Dict]:
    """Return only the top N recommendations."""
    return recommend_models(profile)[:n]


def generate_recommendation_summary(profile: Dict) -> Dict:
    """
    Generate a complete recommendation summary for the UI.

    Returns:
        dict with:
            profile: Dataset profile
            all_recommendations: All ranked recommendations
            top_recommendations: Top 3 recommendations
            dataset_insights: List of human-readable dataset insights
    """
    recommendations = recommend_models(profile)
    top_recs = recommendations[:3]

    # Generate dataset insights
    insights = []

    if profile['is_small_dataset']:
        insights.append({
            'icon': '⚠️',
            'type': 'warning',
            'text': f"Small dataset ({profile['n_rows']:,} rows). Simple models may generalize better.",
        })
    elif profile['is_very_large_dataset']:
        insights.append({
            'icon': '⚡',
            'type': 'info',
            'text': f"Large dataset ({profile['n_rows']:,} rows). Prefer fast models or use sampling.",
        })
    else:
        insights.append({
            'icon': '✅',
            'type': 'success',
            'text': f"Good dataset size ({profile['n_rows']:,} rows, {profile['n_features']} features).",
        })

    if profile['missing_pct'] > 20:
        insights.append({
            'icon': '⚠️',
            'type': 'warning',
            'text': f"High missing value rate ({profile['missing_pct']}%). Ensure imputation is applied.",
        })

    if profile['task_type'] == 'classification' and profile['has_class_imbalance']:
        insights.append({
            'icon': '⚠️',
            'type': 'warning',
            'text': f"Class imbalance: {profile['imbalance_ratio']}:1 ratio. Consider resampling.",
        })

    if profile['is_high_dimensional']:
        insights.append({
            'icon': 'ℹ️',
            'type': 'info',
            'text': f"High-dimensional data ({profile['n_features']} features). Feature selection recommended.",
        })

    if profile['has_mixed_types']:
        insights.append({
            'icon': '🔀',
            'type': 'info',
            'text': f"Mixed feature types detected. Tree-based models handle this natively.",
        })

    return {
        'profile': profile,
        'all_recommendations': recommendations,
        'top_recommendations': top_recs,
        'dataset_insights': insights,
        'task_type': profile['task_type'],
    }
