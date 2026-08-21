from django.urls import path
from . import views
from .label_noise_views import label_noise_detection, auto_fix_labels

urlpatterns = [
    path("clean/", views.clean_data_view, name="clean_data"),
    path("outliers/", views.outlier_detection_view, name="outlier_detection"),
    path("outlier-detection/", views.outlier_detection_view, name="outlier_detection_new"),
    path("label-noise/", label_noise_detection, name="label_noise"),
    path("label-noise/auto-fix/", auto_fix_labels, name="auto_fix_labels"),
    path("irrelevant/", views.irrelevant_feature_view, name="irrelevant_features"),
    path("features/", views.feature_selection_view, name="feature_selection"),
    path("encode-features/", views.encode_features_view, name="encode_features"),
    path("features/get-encoded-data/", views.get_encoded_data_view, name="get_encoded_data"),
    
    # Feature Scaling Page - After Feature Selection
    path("scale-features/", views.scale_features_view, name="scale_features"),

    # ⬇️ Downloads
    path(
        "download-cleaned/",
        views.download_cleaned_dataset,
        name="download_cleaned_dataset"
    ),
    path(
        "download-final/",
        views.download_final_dataset,
        name="download_final_dataset"
    ),
    path(
        "model-training/",
        views.model_training_view,
        name="model_training",
    ),
]
