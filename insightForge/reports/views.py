import pandas as pd
from io import StringIO
from django.shortcuts import render, redirect
from django.contrib import messages

def report_view(request):
    # Required data checks
    if "cleaned_dataset" not in request.session:
        messages.error(request, "Please complete preprocessing first.")
        return redirect("upload_dataset")

    df = pd.read_json(StringIO(request.session["cleaned_dataset"]))

    # Dataset Summary
    dataset_info = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing": int(df.isnull().sum().sum()),
    }

    # Cleaning Report
    cleaning_report = request.session.get("cleaning_report", {})

    # Feature Selection - Get selected features from final_dataset
    selected_features = []
    final_dataset_json = request.session.get("final_dataset", None)
    target_column = request.session.get("target_column", None)
    
    if final_dataset_json:
        try:
            final_df = pd.read_json(StringIO(final_dataset_json))
            # Get all columns from final dataset (features + target)
            all_cols = final_df.columns.tolist()
            # Exclude target column to get only feature columns
            if target_column:
                selected_features = [c for c in all_cols if c != target_column]
            else:
                selected_features = all_cols
        except:
            pass

    # Model Results
    model_report = request.session.get("model_report", {})

    # Label Noise
    noise_report = request.session.get("noise_report", {})

    context = {
        "dataset_info": dataset_info,
        "columns": df.columns.tolist(),
        "preview": df.head(10).to_html(classes="table table-bordered"),

        "cleaning_report": cleaning_report,
        "selected_features": selected_features,
        "model_report": model_report,
        "noise_report": noise_report,
    }

    return render(request, "reports/report.html", context)