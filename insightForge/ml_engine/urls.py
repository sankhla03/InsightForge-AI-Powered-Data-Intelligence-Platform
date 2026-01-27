# ml_engine/urls.py
from django.urls import path
from .views import train_model_view, predict_view, ml_dashboard

urlpatterns = [
    path("", ml_dashboard, name="ml_dashboard"),
    path("train/", train_model_view, name="model_training"),
    path("predict/", predict_view, name="predict"),
]