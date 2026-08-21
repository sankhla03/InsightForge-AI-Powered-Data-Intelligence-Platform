from django.urls import path
from .views import upload_dataset

urlpatterns = [
    path("", upload_dataset, name="upload_dataset"),
]