import time
import pandas as pd
import numpy as np
from django.shortcuts import render, redirect
from django.contrib import messages
from preprocessing.views import get_dataset

from .plotly_utils import (
    histogram_plot,
    box_plot,
    pie_chart,
    pairplot,
)


def _get_latest_processed_dataset(request):
    """
    Load the latest processed dataset from session / disk.
    Priority:
    1. noise_free_dataset (after Label Noise Detection)
    2. outlier_free_dataset (after Outlier Handling)
    3. cleaned_dataset (after Data Cleaning)
    4. dataset (uploaded dataset)
    """
    for key in ["noise_free_dataset", "outlier_free_dataset", "cleaned_dataset", "dataset"]:
        if key in request.session:
            df = get_dataset(request, key, key)
            if df is not None:
                return df
    return None


def visualization_dashboard(request):
    """
    Interactive Visualization Dashboard View.
    Allows users to manually select a chart type (Histogram, Boxplot, Pie Chart, Pairplot),
    configure dynamic fields, generate Plotly charts, and view generated charts.
    """
    df = _get_latest_processed_dataset(request)
    
    if df is None:
        messages.error(request, "No dataset available for visualization. Please upload a dataset first.")
        return redirect("upload_dataset")

    # Mark visualization step completed in session
    request.session["visualization_completed"] = True

    # Identify Column Types
    all_cols = df.columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Categorical columns (object/string/category or low-cardinality discrete integers)
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    target_col = request.session.get("target_column") or request.session.get("noise_detection_target")
    
    for col in df.columns:
        if col not in categorical_cols:
            # Include discrete integer columns with <= 20 unique values in categorical options
            if pd.api.types.is_integer_dtype(df[col]) and df[col].nunique() <= 20:
                categorical_cols.append(col)

    # Saved form selections for persistence
    saved_chart_type = request.session.get("viz_chart_type", "")
    saved_num_col = request.session.get("viz_num_col", numeric_cols[0] if numeric_cols else "")
    saved_cat_col = request.session.get("viz_cat_col", categorical_cols[0] if categorical_cols else "")
    saved_hue = request.session.get("viz_hue", "")
    saved_bins = request.session.get("viz_bins", "20")

    plot_html = None
    chart_title = None

    if request.method == "POST":
        action = request.POST.get("action")
        
        if action == "clear_charts":
            if "generated_charts" in request.session:
                del request.session["generated_charts"]
            messages.info(request, "Generated charts cleared.")
            return redirect("dashboard")

        chart_type = request.POST.get("chart_type")
        num_col = request.POST.get("num_col")
        cat_col = request.POST.get("cat_col")
        hue = request.POST.get("hue")
        bins = request.POST.get("bins", "20")

        # Save for form persistence
        request.session["viz_chart_type"] = chart_type
        request.session["viz_num_col"] = num_col
        request.session["viz_cat_col"] = cat_col
        request.session["viz_hue"] = hue
        request.session["viz_bins"] = bins

        saved_chart_type = chart_type
        saved_num_col = num_col
        saved_cat_col = cat_col
        saved_hue = hue
        saved_bins = bins

        try:
            if chart_type == "histogram":
                if not num_col or num_col not in df.columns:
                    messages.error(request, "Please select a valid numerical column for Histogram.")
                else:
                    chart_title = f"Histogram: {num_col}"
                    plot_html = histogram_plot(df, num_col, bins=bins, title=chart_title)

            elif chart_type == "boxplot":
                if not num_col or num_col not in df.columns:
                    messages.error(request, "Please select a valid numerical column for Boxplot.")
                else:
                    category_col = cat_col if (cat_col and cat_col in df.columns) else None
                    chart_title = f"Boxplot: {num_col}" + (f" by {category_col}" if category_col else "")
                    plot_html = box_plot(df, num_col, category_col=category_col, title=chart_title)

            elif chart_type == "pie":
                target_or_cat = cat_col if (cat_col and cat_col in df.columns) else (categorical_cols[0] if categorical_cols else None)
                if not target_or_cat or target_or_cat not in df.columns:
                    messages.error(request, "Please select a valid category column for Pie Chart.")
                else:
                    chart_title = f"Distribution of {target_or_cat}"
                    plot_html = pie_chart(df, target_or_cat, title=chart_title)

            elif chart_type == "pairplot":
                hue_value = hue if (hue and hue in df.columns) else None
                chart_title = "Pairplot: Relationships Between Numeric Variables"
                plot_html = pairplot(df, numeric_cols=numeric_cols, hue_col=hue_value)

            if plot_html and chart_title:
                # Add to generated charts session history
                generated_charts = request.session.get("generated_charts", [])
                
                chart_entry = {
                    "type": chart_type,
                    "title": chart_title,
                    "plot": plot_html,
                    "timestamp": int(time.time())
                }
                
                # Prepend latest generated chart
                generated_charts.insert(0, chart_entry)
                request.session["generated_charts"] = generated_charts[:10]  # Store last 10 charts
                messages.success(request, f"Generated {chart_title}.")

        except Exception as e:
            messages.error(request, f"Error generating chart: {str(e)}")

    # Retrieve generated charts from session
    generated_charts = request.session.get("generated_charts", [])

    response = render(
        request,
        "visualization/dashboard.html",
        {
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "all_cols": all_cols,
            "plot": plot_html,
            "chart_title": chart_title,
            "generated_charts": generated_charts,
            "saved_chart_type": saved_chart_type,
            "saved_num_col": saved_num_col,
            "saved_cat_col": saved_cat_col,
            "saved_hue": saved_hue,
            "saved_bins": saved_bins,
            "timestamp": int(time.time()),
        },
    )

    response['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response['Pragma'] = 'no-cache'
    response['Expires'] = '0'
    return response
