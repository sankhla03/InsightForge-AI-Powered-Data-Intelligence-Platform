from django.urls import path
from .views import report_view, download_report_pdf, saved_reports_view

urlpatterns = [
    path("generate/", report_view, name="generate_report"),
    path("download-pdf/", download_report_pdf, name="download_report_pdf"),
    path("saved/", saved_reports_view, name="saved_reports"),
]
