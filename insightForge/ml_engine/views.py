import os
import numpy as np
import pandas as pd
import joblib
from io import StringIO

# Classification metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)

# For label binarization
from sklearn.preprocessing import LabelBinarizer

# Regression metrics
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from django.shortcuts import render, redirect
from django.conf import settings
from django.contrib import messages

from sklearn.model_selection import train_test_split

# Classification models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# Regression models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

# For encoding categorical features
from sklearn.preprocessing import OrdinalEncoder

from accounts.utils import require_login


# =====================================================
# HELPER: ENCODE CATEGORICAL FEATURES (STRICT - SKIPS TARGET)
# =====================================================
def encode_features_strict(X_train, X_test, target_column=None):
    """
    Encode categorical/object FEATURES into int64 while STRICTLY skipping the target column.
    
    Pipeline Order:
    1. Target selection (already done before this function)
    2. X / y split (already done before this function)
    3. Encode categorical FEATURES ONLY (this function)
    
    Requirements:
    - Target column is already selected and removed from X
    - Encode ONLY feature columns (X_train, X_test)
    - Do NOT encode or modify target column
    - Convert encoded features into int64
    - Handle multiple categorical columns safely
    - Ignore numeric columns
    - Ensure no data leakage (fit on train, transform on test)
    - Use sklearn only
    
    Encoding Rules:
    - Use OrdinalEncoder for categorical/object features
    - Set handle_unknown='use_encoded_value'
    - Use unknown_value = -1
    - Output dtype must be int64
    
    Args:
        X_train: Training features DataFrame (target already removed)
        X_test: Test features DataFrame (target already removed)
        target_column: Name of target column (for safety check only - not encoded)
    
    Returns:
        X_train_encoded: Encoded training features (int64)
        X_test_encoded: Encoded test features (int64)
        encoders: Dictionary of fitted OrdinalEncoders for each categorical column
    """
    # Safety check: Ensure target column is not in features
    # If target_column is provided and exists in X, remove it (shouldn't happen, but safety first)
    if target_column and target_column in X_train.columns:
        raise ValueError(
            f"CRITICAL ERROR: Target column '{target_column}' should NOT be in features (X). "
            f"Target must be separated BEFORE calling this function."
        )
    
    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()
    encoders = {}
    
    # Identify categorical columns (object or category dtype)
    categorical_cols = X_train_encoded.select_dtypes(
        include=['object', 'category', 'string']
    ).columns.tolist()
    
    numeric_cols = X_train_encoded.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    
    print("\n" + "="*70)
    print("ENCODING CATEGORICAL FEATURES (TARGET STRICTLY SKIPPED)")
    print("="*70)
    print(f"\n[BEFORE ENCODING] Feature dtypes:")
    print("-" * 50)
    
    for col in X_train_encoded.columns:
        col_type = X_train_encoded[col].dtype
        is_categorical = col in categorical_cols
        print(f"  {col}: {col_type} {'[CAT - TO ENCODE]' if is_categorical else '[NUM - SKIP]'}")
    
    print(f"\n  Total features: {len(X_train_encoded.columns)}")
    print(f"  Categorical (to encode): {len(categorical_cols)}")
    print(f"  Numeric (skip): {len(numeric_cols)}")
    print(f"  Target column: {target_column if target_column else 'NOT PROVIDED (already separated)'}")
    
    # Encode each categorical column using OrdinalEncoder
    # IMPORTANT: Fit ONLY on training data to prevent data leakage
    for col in categorical_cols:
        print(f"\n  Encoding: {col}")
        
        # Create OrdinalEncoder with specified settings
        encoder = OrdinalEncoder(
            handle_unknown='use_encoded_value',  # Handle unseen categories in test
            unknown_value=-1,  # Encode unknown as -1
            encoded_missing_value=-1  # Handle missing values
        )
        
        # Fit ONLY on training data (prevent data leakage)
        encoder.fit(X_train_encoded[[col]])
        
        # Transform both train and test
        X_train_encoded[col] = encoder.transform(X_train_encoded[[col]])
        X_test_encoded[col] = encoder.transform(X_test_encoded[[col]])
        
        # Store encoder for potential inverse transform later
        encoders[col] = encoder
        
        print(f"    Categories: {len(encoder.categories_[0])} unique values")
        print(f"    Encoded range: [{int(X_train_encoded[col].min())}, {int(X_train_encoded[col].max())}]")
    
    # Convert ALL encoded columns to int64 (required output dtype)
    for col in categorical_cols:
        X_train_encoded[col] = X_train_encoded[col].astype('int64')
        X_test_encoded[col] = X_test_encoded[col].astype('int64')
    
    # Convert numeric columns to float64 to handle any potential NaN from preprocessing
    for col in numeric_cols:
        X_train_encoded[col] = X_train_encoded[col].astype('float64')
        X_test_encoded[col] = X_test_encoded[col].astype('float64')
    
    print("\n[AFTER ENCODING] Feature dtypes:")
    print("-" * 50)
    for col in X_train_encoded.columns:
        col_type = X_train_encoded[col].dtype
        is_categorical = col in categorical_cols
        print(f"  {col}: {col_type} {'[ENCODED CAT->int64]' if is_categorical else '[NUMERIC]'}")
    
    print(f"\n  All categorical columns converted to: int64")
    print(f"  All numeric columns converted to: float64")
    print("="*70 + "\n")
    
    return X_train_encoded, X_test_encoded, encoders


# =====================================================
# TASK TYPE DETECTION
# =====================================================
def detect_task_type(y, threshold=10):
    """
    Detect whether the task is classification or regression.
    
    Rules:
    - If target is numeric dtype → Regression (most reliable indicator)
    - If target is non-numeric → Classification (string/category labels)
    - The unique value threshold is only used as a secondary check
    
    Args:
        y: Target variable (pandas Series)
        threshold: Minimum unique values for numeric target to be regression (default: 10)
    
    Returns:
        'classification' or 'regression'
    """
    n_unique = y.nunique()
    
    # PRIMARY CHECK: If target is numeric dtype → Regression
    # This is the most reliable indicator of a regression problem
    if pd.api.types.is_numeric_dtype(y):
        # For numeric targets, use unique value threshold
        if n_unique >= threshold:
            return 'regression'
        else:
            # Numeric but very few unique values (e.g., binned data)
            # Check if values are truly continuous or discretized
            # If unique values < threshold but dtype is float, still treat as regression
            if pd.api.types.is_float_dtype(y):
                return 'regression'
            # If unique values < threshold and dtype is int, could be either
            # Default to classification for discrete numeric targets
            return 'classification'
    
    # SECONDARY: If target is non-numeric → Classification
    return 'classification'


# =====================================================
# DASHBOARD
# =====================================================
def ml_dashboard(request):
    if not require_login(request):
        return redirect("login")

    return render(request, "ml_engine/dashboard.html")


# =====================================================
# TRAIN MODEL
# URL: /ml/train/
# =====================================================
def train_model_view(request):
    if "final_dataset" not in request.session:
        messages.error(request, "Complete feature selection first.")
        return redirect("feature_selection")

    # Load dataset from session - use StringIO to handle large JSON strings
    try:
        df = pd.read_json(
            StringIO(request.session["final_dataset"]),
            orient="columns"
        )
    except Exception as e:
        messages.error(request, f"Error loading dataset: {str(e)}")
        return redirect("feature_selection")
    
    columns = df.columns.tolist()
    
    # Get saved target column from feature selection
    saved_target = request.session.get("target_column", "")
    # Validate saved target exists in current columns
    if saved_target and saved_target not in df.columns:
        saved_target = ""

    # GET → show target dropdown with saved target pre-selected
    if request.method == "GET":
        return render(request, "ml_engine/train.html", {
            "columns": columns,
            "saved_target": saved_target,
        })

    # POST → train models
    target = request.POST.get("target")

    if target not in df.columns:
        messages.error(request, "Invalid target selected.")
        return redirect("train_model")

    X = df.drop(columns=[target])
    y = df[target]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Detect task type
    task_type = detect_task_type(y)
    n_classes = y.nunique() if task_type == 'classification' else None
    
    # DEBUG: Log task type detection
    print("\n" + "=" * 70)
    print("TASK TYPE DETECTION")
    print("=" * 70)
    print(f"Target column: '{target}'")
    print(f"Target dtype: {y.dtype}")
    print(f"Unique values: {y.nunique()}")
    print(f"Samples: {len(y)}")
    print(f"DETECTED TASK TYPE: {task_type}")
    print("=" * 70 + "\n")
    
    # Force fresh detection - don't use cached value
    # Clear old session data to ensure fresh detection
    if "task_type" in request.session:
        del request.session["task_type"]

    # =========================================
    # CLASSIFICATION PIPELINE
    # =========================================
    if task_type == 'classification':
        results = train_classification_models(
            X_train, X_test, y_train, y_test, target, X.columns.tolist(), n_classes, request
        )
    
    # =========================================
    # REGRESSION PIPELINE
    # =========================================
    else:
        results = train_regression_models(
            X_train, X_test, y_train, y_test, target, X.columns.tolist(), request
        )

    return render(request, "ml_engine/train.html", {
        "columns": columns,
        "saved_target": saved_target,
        "results": results,
    })


# =====================================================
# CLASSIFICATION TRAINING
# =====================================================
def train_classification_models(X_train, X_test, y_train, y_test, target, feature_columns, n_classes, request):
    """Train and evaluate classification models."""
    
    # Encode categorical features ONLY (strictly skips target column)
    # Target is already separated, so X_train/X_test contain only features
    X_train_encoded, X_test_encoded, encoders = encode_features_strict(
        X_train, X_test, target_column=target
    )
    
    # Define classification models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "Naive Bayes": GaussianNB(),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    # Train and evaluate all models
    model_results = {}
    best_model_name = None
    best_f1 = 0

    for name, model in models.items():
        # Train
        model.fit(X_train_encoded, y_train)
        y_pred = model.predict(X_test_encoded)
        
        # Calculate classification metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        # Calculate ROC-AUC (for multiclass, use one-vs-rest)
        try:
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)
                if n_classes == 2:
                    roc_auc = roc_auc_score(y_test, y_proba[:, 1])
                else:
                    y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))
                    roc_auc = roc_auc_score(y_test_bin, y_proba, multi_class='ovr', average='weighted')
            else:
                roc_auc = 0
        except:
            roc_auc = 0

        model_results[name] = {
            "accuracy": round(acc * 100, 2),
            "precision": round(prec * 100, 2),
            "recall": round(rec * 100, 2),
            "f1": round(f1 * 100, 2),
            "roc_auc": round(roc_auc * 100, 2) if roc_auc > 0 else 0,
            "confusion_matrix": cm.tolist(),
            "model": model,
        }

        # Track best model by F1-score
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name

    # Get best model details
    best_result = model_results[best_model_name]
    best_model = best_result["model"]

    # Save best model to disk
    model_dir = os.path.join(settings.BASE_DIR, "saved_models")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "model.pkl")
    joblib.dump(best_model, model_path)

    # Save for prediction
    request.session["model_path"] = model_path
    request.session["model_features"] = feature_columns
    request.session["model_target"] = target
    request.session["model_is_classification"] = True

    # Save for report generator
    request.session["model_report"] = {
        "best_model": best_model_name,
        "accuracy": best_result["accuracy"],
        "task_type": "classification",
    }

    # Get class labels
    class_labels = sorted(y_test.unique().tolist())

    # Generate confusion matrix heatmap for best model
    cm_image_path = None
    cm_metrics = None
    if best_result["confusion_matrix"]:
        cm_image_path = generate_confusion_matrix_plot(
            best_result["confusion_matrix"], class_labels, best_model_name, request
        )
        
        # Calculate TP, FP, FN, TN for each class
        cm_metrics = calculate_cm_metrics(best_result["confusion_matrix"], class_labels)

    # Convert model_results to list for simpler template rendering
    model_results_list = []
    for name, data in model_results.items():
        model_results_list.append({
            "name": name,
            "accuracy": data["accuracy"],
            "precision": data["precision"],
            "recall": data["recall"],
            "f1": data["f1"],
            "roc_auc": data["roc_auc"],
            "is_best": name == best_model_name,
        })

    return {
        "target": target,
        "task_type": "classification",
        "model_results_list": model_results_list,
        "best_model": best_model_name,
        "best_accuracy": best_result["accuracy"],
        "best_precision": best_result["precision"],
        "best_recall": best_result["recall"],
        "best_f1": best_result["f1"],
        "best_roc_auc": best_result["roc_auc"],
        "best_confusion_matrix": best_result["confusion_matrix"],
        "cm_image_path": cm_image_path,
        "cm_metrics": cm_metrics,
        "features": feature_columns,
        "class_labels": class_labels,
        "n_classes": n_classes,
    }


# =====================================================
# REGRESSION TRAINING
# =====================================================
def train_regression_models(X_train, X_test, y_train, y_test, target, feature_columns, request):
    """Train and evaluate regression models."""
    
    # Encode categorical features ONLY (strictly skips target column)
    # Target is already separated, so X_train/X_test contain only features
    X_train_encoded, X_test_encoded, encoders = encode_features_strict(
        X_train, X_test, target_column=target
    )
    
    # Define regression models
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=200, random_state=42),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "Support Vector Regressor (SVR)": SVR(),
    }

    # Train and evaluate all models
    model_results = {}
    best_model_name = None
    best_r2 = -np.inf

    for name, model in models.items():
        # Train
        model.fit(X_train_encoded, y_train)
        y_pred = model.predict(X_test_encoded)
        
        # Calculate regression metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Calculate accuracy as percentage of predictions within 10% tolerance
        # For each prediction, check if it falls within ±10% of actual value
        tolerance = 0.10  # 10% tolerance
        y_test_arr = y_test.values
        within_tolerance = np.abs(y_pred - y_test_arr) <= (np.abs(y_test_arr) * tolerance)
        # Handle zero actual values - use fixed tolerance
        zero_mask = y_test_arr == 0
        if np.any(zero_mask):
            within_tolerance[zero_mask] = np.abs(y_pred[zero_mask]) <= tolerance
        accuracy_pct = (within_tolerance.sum() / len(within_tolerance)) * 100
        
        model_results[name] = {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "r2": round(r2 * 100, 2),
            "accuracy": round(accuracy_pct, 2),
            "model": model,
            "y_pred": y_pred,
        }

        # Track best model by R² score
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name

    # Get best model details
    best_result = model_results[best_model_name]
    best_model = best_result["model"]

    # Save best model to disk
    model_dir = os.path.join(settings.BASE_DIR, "saved_models")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "model.pkl")
    joblib.dump(best_model, model_path)

    # Save for prediction
    request.session["model_path"] = model_path
    request.session["model_features"] = feature_columns
    request.session["model_target"] = target
    request.session["model_is_classification"] = False

    # Save for report generator
    request.session["model_report"] = {
        "best_model": best_model_name,
        "r2": best_result["r2"],
        "mae": best_result["mae"],
        "rmse": best_result["rmse"],
        "task_type": "regression",
    }

    # Generate regression plots
    avp_image_path = None  # Actual vs Predicted
    residuals_image_path = None  # Residuals distribution
    
    avp_image_path = generate_actual_vs_predicted_plot(
        y_test.values, best_result["y_pred"], best_model_name, request
    )
    residuals_image_path = generate_residuals_plot(
        y_test.values, best_result["y_pred"], best_model_name, request
    )

    # Convert model_results to list for simpler template rendering
    model_results_list = []
    for name, data in model_results.items():
        model_results_list.append({
            "name": name,
            "mae": data["mae"],
            "rmse": data["rmse"],
            "r2": data["r2"],
            "accuracy": data["accuracy"],
            "is_best": name == best_model_name,
        })

    # Calculate feature importance if available
    feature_importance = None
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = list(zip(feature_columns, best_model.feature_importances_))
        feature_importance = sorted(feature_importance, key=lambda x: x[1], reverse=True)

    return {
        "target": target,
        "task_type": "regression",
        "model_results_list": model_results_list,
        "best_model": best_model_name,
        "best_mae": best_result["mae"],
        "best_rmse": best_result["rmse"],
        "best_r2": best_result["r2"],
        "best_accuracy": best_result["accuracy"],
        "avp_image_path": avp_image_path,
        "residuals_image_path": residuals_image_path,
        "features": feature_columns,
        "feature_importance": feature_importance,
        "y_test_min": round(y_test.min(), 2),
        "y_test_max": round(y_test.max(), 2),
    }


# =====================================================
# PLOTTING FUNCTIONS
# =====================================================
def generate_confusion_matrix_plot(cm_array, class_labels, model_name, request):
    """Generate and save confusion matrix heatmap."""
    cm_dir = os.path.join(settings.BASE_DIR, "static", "images")
    os.makedirs(cm_dir, exist_ok=True)
    cm_image_path = os.path.join(cm_dir, f"confusion_matrix_{request.session.session_key or 'default'}.png")
    
    plt.figure(figsize=(10, 8))
    
    # Create heatmap with annotations
    sns.heatmap(
        cm_array,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_labels,
        yticklabels=class_labels,
        linewidths=2,
        linecolor='white',
        cbar_kws={'label': 'Count'},
        annot_kws={'size': 14, 'weight': 'bold'}
    )
    plt.title(f'Confusion Matrix - {model_name}\n(Multi-Class Classification)', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('Actual Label', fontsize=12, fontweight='bold')
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11, rotation=0)
    plt.tight_layout()
    plt.savefig(cm_image_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Make path relative for web
    return f"/static/images/{os.path.basename(cm_image_path)}"


def calculate_cm_metrics(cm_array, class_labels):
    """Calculate TP, FP, FN, TN for each class."""
    cm_metrics = []
    cm_array_np = np.array(cm_array)
    
    for i in range(len(class_labels)):
        tp = int(cm_array_np[i, i])
        fp = int(sum(cm_array_np[:, i]) - tp)
        fn = int(sum(cm_array_np[i, :]) - tp)
        total = sum(sum(cm_array_np))
        tn = total - tp - fp - fn
        cm_metrics.append({
            "class": class_labels[i],
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn
        })
    
    return cm_metrics


def generate_actual_vs_predicted_plot(y_actual, y_pred, model_name, request):
    """Generate Actual vs Predicted scatter plot for regression."""
    cm_dir = os.path.join(settings.BASE_DIR, "static", "images")
    os.makedirs(cm_dir, exist_ok=True)
    avp_image_path = os.path.join(cm_dir, f"actual_vs_predicted_{request.session.session_key or 'default'}.png")
    
    plt.figure(figsize=(10, 8))
    
    # Scatter plot
    plt.scatter(y_actual, y_pred, alpha=0.6, edgecolors='black', linewidth=0.5, c='steelblue')
    
    # Perfect prediction line
    min_val = min(min(y_actual), min(y_pred))
    max_val = max(max(y_actual), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Values', fontsize=12, fontweight='bold')
    plt.ylabel('Predicted Values', fontsize=12, fontweight='bold')
    plt.title(f'Actual vs Predicted - {model_name}\n(Regression)', fontsize=14, fontweight='bold')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(avp_image_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return f"/static/images/{os.path.basename(avp_image_path)}"


def generate_residuals_plot(y_actual, y_pred, model_name, request):
    """Generate Residuals distribution plot for regression."""
    cm_dir = os.path.join(settings.BASE_DIR, "static", "images")
    os.makedirs(cm_dir, exist_ok=True)
    residuals_image_path = os.path.join(cm_dir, f"residuals_{request.session.session_key or 'default'}.png")
    
    residuals = y_actual - y_pred
    
    plt.figure(figsize=(10, 8))
    
    # Histogram with KDE
    sns.histplot(residuals, kde=True, color='steelblue', edgecolor='black', alpha=0.7)
    
    plt.xlabel('Residuals (Actual - Predicted)', fontsize=12, fontweight='bold')
    plt.ylabel('Frequency', fontsize=12, fontweight='bold')
    plt.title(f'Residuals Distribution - {model_name}\n(Regression)', fontsize=14, fontweight='bold')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Residual')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(residuals_image_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return f"/static/images/{os.path.basename(residuals_image_path)}"


# =====================================================
# PREDICTION
# URL: /ml/predict/
# =====================================================
def predict_view(request):
    if "model_path" not in request.session:
        messages.error(request, "Model not trained yet.")
        return redirect("train_model")

    if "model_features" not in request.session:
        messages.error(request, "Feature metadata missing.")
        return redirect("train_model")

    model_path = request.session["model_path"]
    feature_columns = request.session["model_features"]
    is_classification = request.session.get("model_is_classification", True)

    if not os.path.exists(model_path):
        messages.error(request, "Saved model file not found.")
        return redirect("train_model")

    prediction = None
    probability = None
    prediction_label = "Prediction"

    if request.method == "POST":
        input_data = {}

        try:
            for col in feature_columns:
                input_data[col] = float(request.POST.get(col))
        except (TypeError, ValueError):
            messages.error(request, "Invalid input values.")
            return redirect("predict")

        input_df = pd.DataFrame([input_data])
        model = joblib.load(model_path)

        prediction = model.predict(input_df)[0]

        if is_classification:
            prediction_label = "Predicted Class"
            if hasattr(model, "predict_proba"):
                probability = round(
                    max(model.predict_proba(input_df)[0]) * 100, 2
                )
        else:
            prediction_label = "Predicted Value"
            # Format prediction for regression
            prediction = round(float(prediction), 4)

    return render(request, "ml_engine/predict.html", {
        "features": feature_columns,
        "prediction": prediction,
        "probability": probability,
        "prediction_label": prediction_label,
        "is_classification": is_classification,
    })

