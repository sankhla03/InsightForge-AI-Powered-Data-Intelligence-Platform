# ml_engine/urls.py
from django.urls import path
from .views import train_model_view, predict_view, ml_dashboard
from .automl_views import automl_view, automl_status_view

urlpatterns = [
    path("", ml_dashboard, name="ml_dashboard"),
    path("train/", train_model_view, name="model_training"),
    path("predict/", predict_view, name="predict"),

    # AutoML routes
    path("automl/", automl_view, name="automl"),
    path("automl/status/", automl_status_view, name="automl_status"),
]