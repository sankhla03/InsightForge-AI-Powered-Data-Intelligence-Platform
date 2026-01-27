import pandas as pd
from io import StringIO
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.contrib import messages
from .plotly_utils import (
    histogram_plot,
    box_plot,
    scatter_plot,
    pie_chart,
    correlation_heatmap,
    pairplot,
)
import time


def visualization_dashboard(request):
    # Check for datasets in order of processing (most processed first)
    if "final_dataset" in request.session:
        df = pd.read_json(StringIO(request.session["final_dataset"]))
    elif "noise_free_dataset" in request.session:
        df = pd.read_json(StringIO(request.session["noise_free_dataset"]))
    elif "outlier_free_dataset" in request.session:
        df = pd.read_json(StringIO(request.session["outlier_free_dataset"]))
    elif "cleaned_dataset" in request.session:
        df = pd.read_json(StringIO(request.session["cleaned_dataset"]))
    elif "dataset" in request.session:
        df = pd.read_json(StringIO(request.session["dataset"]))
    else:
        messages.error(request, "No dataset available for visualization.")
        return redirect("upload_dataset")

    # Get columns from current dataframe - this ensures fresh data every time
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    all_cols = df.columns.tolist()
    
    # Store column count in session to detect dataset changes
    current_col_count = len(all_cols)
    session_col_count = request.session.get('dataset_col_count', 0)
    
    # If column count changed, clear any cached form selections
    if current_col_count != session_col_count:
        request.session['dataset_col_count'] = current_col_count
        # Clear cached column selections
        if 'viz_column1' in request.session:
            del request.session['viz_column1']
        if 'viz_column2' in request.session:
            del request.session['viz_column2']
        if 'viz_hue' in request.session:
            del request.session['viz_hue']

    plot_html = None
    
    # Get saved selections for form persistence
    saved_column1 = request.session.get('viz_column1', '')
    saved_column2 = request.session.get('viz_column2', '')
    saved_hue = request.session.get('viz_hue', '')

    if request.method == "POST":
        chart_type = request.POST.get("chart_type")
        col1 = request.POST.get("column1")
        col2 = request.POST.get("column2")
        hue = request.POST.get("hue")

        # Save selections for form persistence
        request.session['viz_column1'] = col1
        request.session['viz_column2'] = col2
        request.session['viz_hue'] = hue
        
        saved_column1 = col1
        saved_column2 = col2
        saved_hue = hue

        try:
            if chart_type == "histogram":
                plot_html = histogram_plot(df, col1)

            elif chart_type == "boxplot":
                plot_html = box_plot(df, col1)

            elif chart_type == "scatter":
                hue_value = hue if hue != "" else None
                plot_html = scatter_plot(df, col1, col2, hue_value)

            elif chart_type == "pie":
                plot_html = pie_chart(df, col1)

            elif chart_type == "correlation":
                plot_html = correlation_heatmap(df)

            elif chart_type == "pairplot":
                hue_value = hue if hue != "" else None
                plot_html = pairplot(df, hue_col=hue_value)

        except Exception as e:
            messages.error(request, f"Visualization error: {e}")

    response = render(
        request,
        "visualization/dashboard.html",
        {
            "numeric_cols": numeric_cols,
            "all_cols": all_cols,
            "plot": plot_html,
            "saved_column1": saved_column1,
            "saved_column2": saved_column2,
            "saved_hue": saved_hue,
            "timestamp": int(time.time()),  # Cache busting
        },
    )
    
    # Prevent browser caching to ensure fresh data
    response['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response['Pragma'] = 'no-cache'
    response['Expires'] = '0'
    
    return response
