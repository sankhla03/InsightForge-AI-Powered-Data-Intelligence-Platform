from django.urls import path
from .views import report_view

urlpatterns = [
    path("generate/", report_view, name="generate_report"),
]