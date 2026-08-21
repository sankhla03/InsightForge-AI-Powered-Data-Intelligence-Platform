"""
Comprehensive ML Pipeline Report Generator

This module generates detailed reports for the InsightForge ML pipeline,
covering all stages from data upload to model prediction.
"""

import pandas as pd
import numpy as np
import json
import os
from io import StringIO
from django.conf import settings


class PipelineReportGenerator:
    """
    Generates comprehensive reports for the ML pipeline.
    
    Consumes existing session data WITHOUT modifying any pipeline state.
    """
    
    def __init__(self, request):
        self.request = request
        self.session = request.session
        
    # ========================================================================
    # 1. PROJECT OVERVIEW
    # ========================================================================
    def get_project_overview(self):
        """Get project overview information."""
        overview = {
            "project_name": "InsightForge ML Pipeline",
            "project_description": (
                "An end-to-end machine learning pipeline for automated data preprocessing, "
                "feature engineering, model training, and prediction. The pipeline includes "
                "data cleaning, outlier detection, label noise detection, feature selection, "
                "feature scaling, model comparison, and prediction capabilities."
            ),
            "dataset_name": None,
            "dataset_upload_time": None,
            "total_pipeline_stages": 8,
            "stages_completed": 0,
        }
        
        # Count completed stages
        stages = [
            ("dataset", "Data Upload"),
            ("cleaned_dataset", "Data Cleaning"),
            ("outlier_free_dataset", "Outlier Handling"),
            ("noise_free_dataset", "Label Noise Detection"),
            ("final_dataset", "Feature Selection"),
            ("scaled_dataset", "Feature Scaling"),
            ("model_report", "Model Training"),
            ("model_path", "Model Ready"),
        ]
        
        completed = []
        for key, name in stages:
            if key in self.session or (
                key == "model_path" and 
                self.session.get("model_path") and 
                os.path.exists(self.session.get("model_path"))
            ):
                completed.append(name)
        
        overview["stages_completed"] = len(completed)
        overview["pipeline_completeness"] = round(
            len(completed) / len(stages) * 100, 1
        )
        
        # Get dataset info if available
        if "dataset" in self.session:
            try:
                df = pd.read_json(
                    StringIO(self.session["dataset"]),
                    orient="columns"
                )
                overview["dataset_name"] = self.session.get("dataset_name", "Uploaded Dataset")
                overview["dataset_rows"] = len(df)
                overview["dataset_columns"] = len(df.columns)
            except:
                pass
        
        return overview
    
    # ========================================================================
    # 2. DATASET QUALITY SUMMARY
    # ========================================================================
    def get_dataset_quality_summary(self):
        """Get dataset quality metrics after cleaning."""
        summary = {
            "total_rows": 0,
            "total_columns": 0,
            "duplicate_rows": 0,
            "missing_values": 0,
            "clean_rows_retained": 0,
            "cleaning_performed": False,
        }
        
        # Get original dataset info
        if "dataset" in self.session:
            try:
                df = pd.read_json(
                    StringIO(self.session["dataset"]),
                    orient="columns"
                )
                summary["total_rows"] = len(df)
                summary["total_columns"] = len(df.columns)
                summary["original_rows"] = len(df)
            except:
                pass
        
        # Get cleaning report
        cleaning_report = self.session.get("cleaning_report", {})
        if cleaning_report:
            summary["cleaning_performed"] = True
            summary["duplicate_rows"] = cleaning_report.get("duplicate_rows", 0)
            summary["missing_values_handled"] = cleaning_report.get("missing_handled", 0)
        
        # Get cleaned dataset
        if "cleaned_dataset" in self.session:
            try:
                df = pd.read_json(
                    StringIO(self.session["cleaned_dataset"]),
                    orient="columns"
                )
                summary["clean_rows_retained"] = len(df)
                summary["total_rows"] = len(df)
                summary["total_columns"] = len(df.columns)
            except:
                pass
        
        return summary
    
    # ========================================================================
    # 3. OUTLIER ANALYSIS SUMMARY
    # ========================================================================
    def get_outlier_analysis_summary(self):
        """Get outlier detection and handling summary."""
        summary = {
            "method": "IQR (Interquartile Range)",
            "outliers_detected": 0,
            "rows_corrected": 0,
            "handling_technique": None,
            "features_analyzed": [],
            "features_with_outliers": [],
            "outlier_details": {},
        }
        
        # Get outlier report from session
        outlier_report = self.session.get("outlier_report", {})
        if outlier_report:
            summary["outliers_detected"] = outlier_report.get("outlier_count", 0)
            summary["handling_technique"] = outlier_report.get("method", "winsorized")
            summary["rows_corrected"] = outlier_report.get("rows_corrected", 0)
        
        # Get outlier detection results
        outlier_count = self.session.get("outlier_count", 0)
        if outlier_count > 0:
            summary["outliers_detected"] = outlier_count
            summary["handling_technique"] = self.session.get("outlier_save_method", "winsorized")
        
        # Get current dataset state
        if "outlier_free_dataset" in self.session:
            try:
                df = pd.read_json(
                    StringIO(self.session["outlier_free_dataset"]),
                    orient="columns"
                )
                summary["rows_after_outlier_handling"] = len(df)
            except:
                pass
        
        return summary
    
    # ========================================================================
    # 4. VISUALIZATION SUMMARY
    # ========================================================================
    def get_visualization_summary(self):
        """Get summary of generated visualizations with actual Plotly HTML representations."""
        summary = {
            "visualization_types": [],
            "generated_charts": [],
            "chart_details": [],
            "key_visuals": [],
        }
        
        # Chart type names mapping
        chart_type_names = {
            "histogram": "Histogram",
            "boxplot": "Box Plot",
            "scatter": "Scatter Plot",
            "pie": "Pie Chart",
            "correlation": "Correlation Heatmap",
            "pairplot": "Pair Plot",
        }
        
        # Load dataset for visualization rendering
        from preprocessing.views import get_dataset
        df = get_dataset(self.request, "noise_free_dataset", "noise_free_dataset")
        if df is None:
            df = get_dataset(self.request, "outlier_free_dataset", "outlier_free_dataset")
        if df is None:
            df = get_dataset(self.request, "cleaned_dataset", "cleaned_dataset")
        if df is None:
            df = get_dataset(self.request, "dataset", "dataset")

        key_visuals = []

        if df is not None and len(df) > 0:
            try:
                from visualization.plotly_utils import (
                    correlation_heatmap,
                    pairplot,
                    histogram_plot,
                    pie_chart,
                )
                
                target = self.session.get("target_column")
                numeric_cols = df.select_dtypes(include="number").columns.tolist()

                # 1. Target Distribution
                if target and target in df.columns:
                    if pd.api.types.is_numeric_dtype(df[target]) and df[target].nunique() > 10:
                        t_plot = histogram_plot(df, target)
                    else:
                        t_plot = pie_chart(df, target)
                    
                    key_visuals.append({
                        "title": f"Target Distribution ({target})",
                        "type": "Target Distribution",
                        "description": f"Frequency and value distribution of target variable '{target}' across observations.",
                        "plot_html": t_plot,
                    })

                # 2. Correlation Heatmap
                if len(numeric_cols) >= 2:
                    corr_html = correlation_heatmap(df)
                    key_visuals.append({
                        "title": "Correlation Heatmap",
                        "type": "Correlation Heatmap",
                        "description": "Pairwise linear correlation matrix across numerical feature variables.",
                        "plot_html": corr_html,
                    })

                # 3. Pair Plot
                if len(numeric_cols) >= 2:
                    hue_col = target if (target and target in df.columns and df[target].nunique() <= 10) else None
                    pp_html = pairplot(df, numeric_cols=numeric_cols[:5], hue_col=hue_col)
                    key_visuals.append({
                        "title": "Pair Plot",
                        "type": "Pair Plot",
                        "description": "Scatter matrix displaying pairwise relationships and marginal distributions across numeric features.",
                        "plot_html": pp_html,
                    })

            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Plotly generation in report: {e}")

        summary["key_visuals"] = key_visuals

        # Get tracked generated charts from session
        generated_charts = self.session.get('generated_charts', [])
        
        # If no session data, try file
        if not generated_charts:
            charts_file = os.path.join(settings.BASE_DIR, 'generated_charts', 'charts.json')
            if os.path.exists(charts_file):
                import json
                try:
                    with open(charts_file, 'r') as f:
                        generated_charts = json.load(f)
                except:
                    generated_charts = []
        
        # Format generated charts with details
        chart_details = []
        for i, chart in enumerate(generated_charts):
            chart_type = chart.get("type", "unknown")
            col1 = chart.get("column1", "")
            col2 = chart.get("column2", "")
            
            # Create readable description
            if chart_type == "histogram":
                desc = f"Distribution of '{col1}'"
            elif chart_type == "boxplot":
                desc = f"Box plot of '{col1}'"
            elif chart_type == "scatter":
                desc = f"Scatter: '{col1}' vs '{col2}'"
            elif chart_type == "pie":
                desc = f"Pie chart of '{col1}'"
            elif chart_type == "correlation":
                desc = "Correlation heatmap of all features"
            elif chart_type == "pairplot":
                desc = "Pair plot of all features"
            else:
                desc = f"{chart_type} chart"
            
            chart_details.append({
                "number": i + 1,
                "type": chart_type_names.get(chart_type, chart_type.title()),
                "description": desc,
                "timestamp": chart.get("timestamp", 0),
            })
        
        # Get image files from static directory
        static_dir = os.path.join(settings.BASE_DIR, "static", "images")

        # Existing image metadata (kept for backward compatibility)
        existing_images = []

        # Unified gallery metadata (used by enhanced template)
        gallery_images = []

        def _safe_title(s: str) -> str:
            return (s or "").replace("_", " ").strip().title()

        # Map common filename prefixes to visualization types
        filename_type_map = {
            # Dataset / EDA
            "histogram": "Histogram",
            "box": "Box Plot",
            "boxplot": "Box Plot",
            "scatter": "Scatter Plot",
            "pair": "Pair Plot",
            "pairplot": "Pair Plot",
            "heatmap": "Correlation Heatmap",
            "correlation": "Correlation Heatmap",
            "corr": "Correlation Heatmap",
            "count": "Count Plot",
            "countplot": "Count Plot",
            "distribution": "Distribution Plot",
            "dist": "Distribution Plot",
            "missing": "Missing Value Visualization",
            "missing_values": "Missing Value Visualization",
            "outlier": "Outlier Visualization",
            "outliers": "Outlier Visualization",
            "feature": "Feature Importance",
            "feature_importance": "Feature Importance",
            "importance": "Feature Importance",
            "line": "Line Chart",
            "bar": "Bar Chart",
            "pie": "Pie Chart",

            # ML evaluation
            "confusion": "Confusion Matrix",
            "confusion_matrix": "Confusion Matrix",
            "residuals": "Residuals",
            "residual": "Residuals",
            "actual_vs_predicted": "Actual vs Predicted",
            "actual": "Actual vs Predicted",
        }

        # Decide section: ML evaluation vs Dataset/EDA.
        ml_eval_types = {
            "Confusion Matrix",
            "Residuals",
            "Residual Plot",
            "Actual vs Predicted",
            "Actual vs Predicted Graph",
            "Actual",
            "Actual vs Predicted Plot",
        }

        if os.path.exists(static_dir):
            for f in os.listdir(static_dir):
                if not f.endswith(".png"):
                    continue

                file_stem = f.replace(".png", "")
                parts = file_stem.split("_")

                # Prefer full prefix (first 2 tokens) when applicable
                prefix2 = "_".join(parts[:2]).strip().lower() if len(parts) >= 2 else ""
                prefix1 = parts[0].strip().lower() if parts else ""

                chart_type = filename_type_map.get(prefix2) or filename_type_map.get(prefix1) or chart_type_names.get(prefix1) or _safe_title(prefix1)

                # Only include graphs that actually have a usable URL
                url = f"/static/images/{f}"

                graph_title = chart_type
                short_insight = {
                    "Histogram": "Distribution of values in the selected feature.",
                    "Box Plot": "Spread and outliers in the selected feature.",
                    "Scatter Plot": "Relationship between two variables.",
                    "Pair Plot": "Pairwise relationships across multiple features.",
                    "Correlation Heatmap": "Correlation structure across features.",
                    "Count Plot": "Frequency distribution for a categorical/label-like feature.",
                    "Distribution Plot": "Value distribution and shape of the selected feature.",
                    "Missing Value Visualization": "Pattern and extent of missing values in the dataset.",
                    "Outlier Visualization": "Outlier presence and spread for relevant numeric features.",
                    "Feature Importance": "Model-driven feature contribution ranking.",
                    "Line Chart": "Trend over an ordering (e.g., time or index).",
                    "Bar Chart": "Comparative magnitude across categories.",
                    "Pie Chart": "Share of the total across categories.",

                    "Confusion Matrix": "Classification performance across predicted vs actual classes.",
                    "Residuals": "Residual distribution for regression model diagnostics.",
                    "Actual vs Predicted": "How close predictions are to actual values.",
                }.get(chart_type, "Generated visualization from the pipeline.")

                section = "ML Model Evaluation" if chart_type in ml_eval_types else "Dataset Visualization"

                existing_images.append({
                    "filename": f,
                    "url": url,
                    "type": chart_type,
                })

                # Each card must have title/type/description/url/section.
                gallery_images.append({
                    "graph_title": graph_title,
                    "type": chart_type,
                    "description": short_insight,
                    "url": url,
                    "section": section,
                    "filename": f,
                })

        summary["generated_charts"] = generated_charts
        summary["chart_details"] = chart_details
        summary["existing_images"] = existing_images
        summary["gallery_images"] = gallery_images
        # Chart count in UI should reflect actual rendered images available.
        summary["chart_count"] = len(gallery_images)


        return summary
    
    # ========================================================================
    # 5. FEATURE SELECTION SUMMARY
    # ========================================================================
    def get_feature_selection_summary(self):
        """Get feature selection results."""
        summary = {
            "method": None,
            "original_features": 0,
            "selected_features": [],
            "selected_features_count": 0,
            "feature_scores": {},
            "target_column": None,
        }
        
        # Get selected features
        selected_features = self.session.get("selected_features", [])
        if selected_features:
            summary["selected_features"] = selected_features
            summary["selected_features_count"] = len(selected_features)
        
        # Get target column
        target_col = self.session.get("target_column", None)
        if target_col:
            summary["target_column"] = target_col
            # Original features = selected + target
            summary["original_features"] = len(selected_features) + (
                1 if target_col not in selected_features else 0
            )
        
        # Get feature scores
        feature_scores = self.session.get("feature_scores", {})
        if feature_scores:
            summary["feature_scores"] = feature_scores
        
        # Get method used
        method = self.session.get("feature_selection_method", None)
        if method:
            method_names = {
                "correlation": "Correlation-based",
                "kbest": "SelectKBest (F-Score)",
                "rfe": "Recursive Feature Elimination",
                "tree": "Tree-based Importance",
                "none": "All Features",
            }
            summary["method"] = method_names.get(method, method)
        
        return summary
    
    # ========================================================================
    # 6. FEATURE SCALING SUMMARY
    # ========================================================================
    def get_feature_scaling_summary(self):
        """Get feature scaling information."""
        summary = {
            "method": None,
            "features_scaled": [],
            "scaling_applied": False,
            "justification": None,
        }
        
        # Get scaler parameters
        scaler_params = self.session.get("scaler_params", {})
        if scaler_params:
            summary["scaling_applied"] = True
            summary["method"] = scaler_params.get("method", "standard").capitalize()
            summary["features_scaled"] = scaler_params.get("feature_cols", [])
            
            # Generate justification
            method = scaler_params.get("method", "standard")
            justifications = {
                "standard": (
                    "StandardScaler (Z-score normalization) is used because it preserves "
                    "the shape of the distribution and is appropriate for algorithms "
                    "assuming Gaussian-distributed features (e.g., SVM, Logistic Regression)."
                ),
                "minmax": (
                    "MinMaxScaler is used to normalize features to a [0, 1] range, "
                    "which is suitable for neural networks and algorithms with "
                    "bounded activation functions."
                ),
                "robust": (
                    "RobustScaler is used because it is robust to outliers, using "
                    "median and IQR instead of mean and std. This is ideal when the "
                    "dataset contains significant outliers."
                ),
            }
            summary["justification"] = justifications.get(method, "Standard scaling applied.")
            
            # Verify scaling was applied to selected features only
            summary["only_selected_features"] = True
            selected_features = self.session.get("selected_features", [])
            if selected_features:
                summary["only_selected_features"] = set(summary["features_scaled"]).issubset(
                    set([f for f in selected_features if f != summary.get("target_column")])
                )
        
        return summary
    
    # ========================================================================
    # 7. MODEL TRAINING & COMPARISON
    # ========================================================================
    def get_model_training_summary(self):
        """Get model training results and comparison."""
        summary = {
            "task_type": None,
            "models_trained": [],
            "comparison_table": [],
            "best_model": None,
            "best_metrics": {},
        }
        
        # Get model report
        model_report = self.session.get("model_report", {})
        if model_report:
            summary["task_type"] = model_report.get("task_type", "classification")
            summary["best_model"] = model_report.get("best_model")
            
            # Get detailed model results from session
            model_results = self.session.get("model_results", {})
            if model_results:
                for name, metrics in model_results.items():
                    row = {
                        "model": name,
                        "is_best": name == summary["best_model"],
                    }
                    if summary["task_type"] == "classification":
                        row["accuracy"] = metrics.get("accuracy", 0)
                        row["precision"] = metrics.get("precision", 0)
                        row["recall"] = metrics.get("recall", 0)
                        row["f1"] = metrics.get("f1", 0)
                        row["roc_auc"] = metrics.get("roc_auc", 0)
                    else:
                        row["r2"] = metrics.get("r2", 0)
                        row["mae"] = metrics.get("mae", 0)
                        row["rmse"] = metrics.get("rmse", 0)
                        row["accuracy"] = metrics.get("accuracy", 0)
                    summary["comparison_table"].append(row)
                    summary["models_trained"].append(name)
        
        return summary
    
    # ========================================================================
    # 8. MODEL SELECTION
    # ========================================================================
    def get_model_selection_summary(self):
        """Get model selection reasoning."""
        summary = {
            "selected_model": None,
            "selection_metric": None,
            "metric_value": None,
            "reason": None,
            "target_column": None,
            "task_type": None,
        }
        
        model_report = self.session.get("model_report", {})
        if model_report:
            summary["selected_model"] = model_report.get("best_model")
            summary["task_type"] = model_report.get("task_type", "classification")
            summary["target_column"] = self.session.get("target_column")
            
            if summary["task_type"] == "classification":
                summary["selection_metric"] = "F1-Score (Weighted)"
                summary["metric_value"] = model_report.get("f1", model_report.get("accuracy", 0))
                summary["reason"] = (
                    f"The {summary['selected_model']} was selected because it achieved the highest "
                    f"F1-Score ({summary['metric_value']:.2f}%) on the test set. "
                    f"F1-Score balances precision and recall, making it ideal for "
                    f"classification tasks where both false positives and false negatives "
                    f"are important to minimize."
                )
            else:
                summary["selection_metric"] = "R² Score"
                summary["metric_value"] = model_report.get("r2", 0)
                summary["reason"] = (
                    f"The {summary['selected_model']} was selected because it achieved the highest "
                    f"R² Score ({summary['metric_value']:.2f}%) on the test set. "
                    f"R² Score indicates the proportion of variance explained by the model, "
                    f"with higher values indicating better fit."
                )
        
        return summary
    
    # ========================================================================
    # 9. PREDICTION SUMMARY
    # ========================================================================
    def get_prediction_summary(self):
        """Get prediction summary if available."""
        summary = {
            "prediction_performed": False,
            "prediction_samples": 0,
            "predictions": [],
            "model_loaded": False,
        }
        
        # Check if model exists
        model_path = self.session.get("model_path")
        if model_path and os.path.exists(model_path):
            summary["model_loaded"] = True
        
        # Get prediction results from session
        prediction_results = self.session.get("prediction_results", [])
        if prediction_results:
            summary["prediction_performed"] = True
            summary["prediction_samples"] = len(prediction_results)
            summary["predictions"] = prediction_results
        
        return summary
    
    # ========================================================================
    # 10. FINAL CONCLUSION
    # ========================================================================
    def get_final_conclusion(self):
        """Generate final conclusion summary."""
        conclusion = {
            "data_quality_improvements": [],
            "model_readiness": "Not Ready",
            "pipeline_completeness": 0,
            "recommendations": [],
            "pipeline_status": "Incomplete",
        }
        
        # Calculate data quality improvements
        cleaning_report = self.session.get("cleaning_report", {})
        if cleaning_report:
            duplicates = cleaning_report.get("duplicate_rows", 0)
            missing = cleaning_report.get("missing_handled", 0)
            if duplicates > 0:
                conclusion["data_quality_improvements"].append(
                    f"Removed {duplicates} duplicate rows"
                )
            if missing > 0:
                conclusion["data_quality_improvements"].append(
                    f"Handled {missing} missing values using mean/mode imputation"
                )
        
        outlier_count = self.session.get("outlier_count", 0)
        if outlier_count > 0:
            conclusion["data_quality_improvements"].append(
                f"Detected and handled {outlier_count} outliers using IQR method"
            )
        
        noise_report = self.session.get("noise_report", {})
        if noise_report:
            noisy_rows = noise_report.get("noisy_rows", 0)
            if noisy_rows > 0:
                conclusion["data_quality_improvements"].append(
                    f"Detected {noisy_rows} potentially mislabeled samples"
                )
        
        # Check pipeline completeness
        stages = [
            ("dataset", "Data Upload"),
            ("cleaned_dataset", "Data Cleaning"),
            ("outlier_free_dataset", "Outlier Handling"),
            ("noise_free_dataset", "Label Noise Detection"),
            ("final_dataset", "Feature Selection"),
            ("scaled_dataset", "Feature Scaling"),
            ("model_report", "Model Training"),
        ]
        
        completed = 0
        for key, name in stages:
            if key in self.session:
                completed += 1
        
        conclusion["pipeline_completeness"] = round(completed / len(stages) * 100, 1)
        
        if completed == len(stages):
            conclusion["pipeline_status"] = "Complete"
            conclusion["model_readiness"] = "Ready for Predictions"
        elif completed >= 4:
            conclusion["pipeline_status"] = "Near Complete"
            conclusion["model_readiness"] = "Almost Ready"
        else:
            conclusion["model_readiness"] = "Not Ready"
            conclusion["recommendations"].append(
                "Complete all preprocessing steps before model training"
            )
        
        # Add recommendations
        if conclusion["pipeline_completeness"] < 100:
            if "scaled_dataset" not in self.session:
                conclusion["recommendations"].append("Apply feature scaling before training")
            if "model_report" not in self.session:
                conclusion["recommendations"].append("Train models to compare performance")
        
        return conclusion
    
    # ========================================================================
    # GENERATE COMPLETE REPORT
    # ========================================================================
    def generate_complete_report(self):
        """Generate the complete pipeline report."""
        return {
            "project_overview": self.get_project_overview(),
            "dataset_quality": self.get_dataset_quality_summary(),
            "outlier_analysis": self.get_outlier_analysis_summary(),
            "visualization_summary": self.get_visualization_summary(),
            "feature_selection": self.get_feature_selection_summary(),
            "feature_scaling": self.get_feature_scaling_summary(),
            "model_training": self.get_model_training_summary(),
            "model_selection": self.get_model_selection_summary(),
            "prediction_summary": self.get_prediction_summary(),
            "conclusion": self.get_final_conclusion(),
        }


def get_pipeline_report(request):
    """Generate a comprehensive report for the ML pipeline.

    Returns dict with all report sections.
    """
    generator = PipelineReportGenerator(request)
    return generator.generate_complete_report()


def format_report_for_html(report):
    """Format the report data for HTML template rendering.

    Note: keep structure backward-compatible; template may rely on existing keys.
    """
    formatted = {}
    
    # Project Overview
    po = report.get("project_overview", {})
    formatted["project"] = {
        "name": po.get("project_name", "InsightForge ML Pipeline"),
        "description": po.get("project_description", ""),
        "stages_completed": f"{po.get('stages_completed', 0)}/{po.get('total_pipeline_stages', 8)}",
        "completeness": po.get("pipeline_completeness", 0),
    }
    
    # Dataset Quality
    dq = report.get("dataset_quality", {})
    formatted["dataset"] = {
        "total_rows": dq.get("total_rows", 0),
        "total_columns": dq.get("total_columns", 0),
        "duplicates": dq.get("duplicate_rows", 0),
        "missing_handled": dq.get("missing_values_handled", 0),
        "clean_rows": dq.get("clean_rows_retained", 0),
        "cleaning_done": dq.get("cleaning_performed", False),
    }
    
    # Outlier Analysis
    oa = report.get("outlier_analysis", {})
    formatted["outliers"] = {
        "method": oa.get("method", "IQR"),
        "detected": oa.get("outliers_detected", 0),
        "handled": oa.get("handling_technique", "Not handled"),
        "rows_corrected": oa.get("rows_corrected", 0),
    }
    
    # Feature Selection
    fs = report.get("feature_selection", {})
    formatted["features"] = {
        "method": fs.get("method", "Not selected"),
        "original_count": fs.get("original_features", 0),
        "selected_count": fs.get("selected_features_count", 0),
        "selected_list": fs.get("selected_features", []),
        "scores": fs.get("feature_scores", {}),
        "target": fs.get("target_column", ""),
    }
    
    # Feature Scaling
    fsc = report.get("feature_scaling", {})
    formatted["scaling"] = {
        "method": fsc.get("method", "Not applied"),
        "applied": fsc.get("scaling_applied", False),
        "features_count": len(fsc.get("features_scaled", [])),
        "justification": fsc.get("justification", ""),
    }
    
    # Model Training
    mt = report.get("model_training", {})
    formatted["models"] = {
        "task_type": mt.get("task_type", "classification"),
        "best_model": mt.get("best_model", "Not trained"),
        "comparison_table": mt.get("comparison_table", []),
        "models_count": len(mt.get("models_trained", [])),
    }
    
    # Model Selection
    ms = report.get("model_selection", {})
    formatted["selection"] = {
        "selected_model": ms.get("selected_model", ""),
        "metric": ms.get("selection_metric", ""),
        "metric_value": ms.get("metric_value", 0),
        "reason": ms.get("reason", ""),
    }
    
    # Prediction
    ps = report.get("prediction_summary", {})
    formatted["prediction"] = {
        "performed": ps.get("prediction_performed", False),
        "samples": ps.get("prediction_samples", 0),
        "model_ready": ps.get("model_loaded", False),
    }
    
    # Conclusion
    con = report.get("conclusion", {})
    formatted["conclusion"] = {
        "status": con.get("pipeline_status", "Incomplete"),
        "readiness": con.get("model_readiness", "Not Ready"),
        "completeness": con.get("pipeline_completeness", 0),
        "improvements": con.get("data_quality_improvements", []),
        "recommendations": con.get("recommendations", []),
    }
    
    # Visualization
    viz = report.get("visualization_summary", {})
    formatted["visualization"] = {
        "types": viz.get("visualization_types", []),
        "charts": viz.get("generated_charts", []),
        "chart_details": viz.get("chart_details", []),
        "key_visuals": viz.get("key_visuals", []),
        "existing_images": viz.get("existing_images", []),
        "chart_count": viz.get("chart_count", 0),
    }
    
    return formatted

