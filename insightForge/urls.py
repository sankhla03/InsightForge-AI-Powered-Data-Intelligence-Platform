from django.contrib import admin
from django.urls import path, include
from django.views.generic import RedirectView
from django.http import HttpResponse
from preprocessing.views import outlier_detection_view

def favicon_view(request):
    return HttpResponse(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><text y=".9em" font-size="90">🔍</text></svg>',
        content_type='image/svg+xml'
    )

urlpatterns = [
    path("admin/", admin.site.urls),
    path("favicon.ico", favicon_view, name="favicon"),
    path("", RedirectView.as_view(url="/accounts/login/", permanent=False)),  # Bug M5 fix: redirect root to login
    path("datasets/", include("datasets.urls")),       # dataset upload
    path("preprocessing/", include("preprocessing.urls")),
    path("outlier-detection/", outlier_detection_view, name="outlier_detection"),
    path("ml/", include("ml_engine.urls")),
    path("visualization/", include("visualization.urls")),
    path("accounts/", include("accounts.urls")),
    path("reports/", include("reports.urls")),
]
