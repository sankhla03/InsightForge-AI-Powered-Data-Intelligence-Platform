import json
import logging
import os

import pandas as pd
from io import StringIO

from django.contrib import messages
from django.http import HttpResponse
from django.shortcuts import render, redirect

from .report_generator import get_pipeline_report, format_report_for_html
from .feature_importance_analysis import generate_report


logger = logging.getLogger(__name__)

# Use our new ReportLab-based PDF generator (pure Python, no system deps)
try:
    from .pdf_generator import generate_pdf_report
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logger.warning("ReportLab not available — PDF generation disabled.")

def feature_importance_report_view(request):
    """
    Display the Feature Importance Analysis Report
    Explains why encoded categorical features may appear more important
    than numerical features in correlation-based analysis.
    """
    # Bug H8 fix: use get_dataset() instead of raw pd.read_json()
    # NOTE: explicit is None checks — `or` raises ValueError on a DataFrame.
    from preprocessing.views import get_dataset

    df = get_dataset(request, "noise_free_dataset", "noise_free_dataset")
    if df is None:
        df = get_dataset(request, "outlier_free_dataset", "outlier_free_dataset")

    # Get feature scores if available in session
    feature_scores = request.session.get("feature_scores", None)

    # Generate the report content
    report_content = generate_report(df, feature_scores)

    return render(request, "reports/feature_importance_report.html", {
        "report_content": report_content,
        "has_dataset": df is not None,
        "dataset_shape": df.shape if df is not None else None,
        "feature_scores": feature_scores,
    })


def report_view(request):
    """
    Comprehensive ML Pipeline Report View.
    """
    # Required data checks - ensure preprocessing is complete
    if "cleaned_dataset" not in request.session:
        messages.error(request, "Please complete preprocessing first.")
        return redirect("upload_dataset")

    # Bug H7 fix: use get_dataset() instead of raw pd.read_json()
    from preprocessing.views import get_dataset

    # Generate comprehensive pipeline report
    raw_report = get_pipeline_report(request)
    formatted_report = format_report_for_html(raw_report)

    # Dataset Preview
    try:
        df = get_dataset(request, "cleaned_dataset", "cleaned_dataset")
        preview = df.head(10).to_html(classes="table table-bordered") if df is not None else "<p>No preview available</p>"
    except Exception:
        df = None
        preview = "<p>No preview available</p>"

    # Bug M11 fix: do not use dir() to test if a local variable was set.
    # Use explicit None check after the try/except block above.
    column_list = list(df.columns) if df is not None else []

    context = {
        "project": formatted_report["project"],
        "dataset": formatted_report["dataset"],
        "outliers": formatted_report["outliers"],
        "visualization": formatted_report["visualization"],
        "features": formatted_report["features"],
        "scaling": formatted_report["scaling"],
        "models": formatted_report["models"],
        "selection": formatted_report["selection"],
        "prediction": formatted_report["prediction"],
        "conclusion": formatted_report["conclusion"],
        "preview": preview,
        "columns": column_list,
    }

    return render(request, "reports/report.html", context)


# =====================================================
# PDF DOWNLOAD VIEW
# =====================================================
def download_report_pdf(request):
    """
    Generate and download the ML Pipeline Report as PDF.
    Uses ReportLab for high-quality, professional PDF generation.
    Saves PDF to permanent location and returns download link.
    """
    from django.conf import settings
    from datetime import datetime
    import json
    
    # Check if preprocessing is complete
    if "cleaned_dataset" not in request.session:
        messages.error(request, "Please complete preprocessing first.")
        return redirect("upload_dataset")
    
    # Create saved_reports directory if it doesn't exist
    reports_dir = os.path.join(settings.BASE_DIR, 'saved_reports')
    os.makedirs(reports_dir, exist_ok=True)
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_name = f"ml_pipeline_report_{timestamp}"
    pdf_path = os.path.join(reports_dir, f"{report_name}.pdf")
    json_path = os.path.join(reports_dir, f"{report_name}.json")
    
    # Generate report data
    raw_report = get_pipeline_report(request)
    
    # Also save JSON for reference
    try:
        with open(json_path, 'w') as f:
            json.dump(raw_report, f, indent=2, default=str)
    except Exception:
        pass
    
    # Check if ReportLab is available
    if not REPORTLAB_AVAILABLE:
        messages.warning(request, "PDF generation requires ReportLab. Install with: pip install reportlab")
        return redirect("generate_report")
    
    # Generate PDF using ReportLab
    try:
        pdf_bytes = generate_pdf_report(raw_report)
        
        # Save to disk
        with open(pdf_path, 'wb') as f:
            f.write(pdf_bytes)
        
        # Return as download response
        response = HttpResponse(pdf_bytes, content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="insightforge_report_{timestamp}.pdf"'
        return response
        
    except Exception as e:
        logger.exception(f"PDF generation failed: {e}")
        messages.error(request, f"PDF generation failed: {str(e)}")
        return redirect("generate_report")





# =====================================================
# SAVED REPORTS VIEW
# =====================================================
def saved_reports_view(request):
    """
    Display list of previously saved reports.
    """
    from django.conf import settings
    from datetime import datetime
    
    reports_dir = os.path.join(settings.BASE_DIR, 'saved_reports')
    
    if not os.path.exists(reports_dir):
        messages.info(request, "No saved reports found.")
        return render(request, "reports/saved_reports.html", {"reports": []})
    
    reports = []
    for filename in os.listdir(reports_dir):
        if filename.endswith('.pdf') or filename.endswith('.json'):
            file_path = os.path.join(reports_dir, filename)
            stat = os.stat(file_path)
            reports.append({
                "filename": filename,
                "path": file_path,
                "size": f"{stat.st_size / 1024:.1f} KB",
                "created": datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d %H:%M:%S"),
                "type": "PDF" if filename.endswith('.pdf') else "JSON",
            })
    
    # Sort by creation time (newest first)
    reports.sort(key=lambda x: x['created'], reverse=True)
    
    return render(request, "reports/saved_reports.html", {"reports": reports})
