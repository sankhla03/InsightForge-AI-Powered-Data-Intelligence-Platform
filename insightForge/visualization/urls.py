from django.urls import path
from .views import visualization_dashboard

urlpatterns = [
    path("", visualization_dashboard, name="visualization_dashboard"),
]