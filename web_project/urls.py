from django.contrib import admin
from django.urls import include, path
from django.conf import settings
from django.conf.urls.static import static
from users import views as user_views

urlpatterns = [
    path('admin/', admin.site.urls),
    #path('', include("profile_analyzer.urls")),
    path('', include('main.urls')),
    path('searches/', include('data_collector.urls')),
    path('users/', include('users.urls'))
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)