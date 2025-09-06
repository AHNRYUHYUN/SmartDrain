from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/accountapp/', include('accountapp.urls')),
    path('maintenance/', include('maintenance.urls')),
]
