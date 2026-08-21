"""
URL Configuration for Label Noise Detection Module
"""

from django.urls import path
from .label_noise_views import label_noise_detection

urlpatterns = [
    path("preprocessing/label-noise/", label_noise_detection, name="label_noise_detection"),
]

