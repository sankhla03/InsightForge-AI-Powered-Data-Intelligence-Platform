from django.contrib import admin
from django.urls import path, include
from django.views.generic import RedirectView

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", RedirectView.as_view(url="/accounts/register/", permanent=False)),  # Redirect root to register
    path("datasets/", include("datasets.urls")),       # dataset upload
    path("preprocessing/", include("preprocessing.urls")),
    path("ml/", include("ml_engine.urls")),
    path("visualization/", include("visualization.urls")),
    path("accounts/", include("accounts.urls")),
    path("reports/", include("reports.urls")),
]
