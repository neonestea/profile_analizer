'''Users urls'''
from django.urls import path
from . import views
from django.contrib.auth import views as auth_views
from users import views as user_views


urlpatterns = [
    #path('', views.search_home, name='search_home'),
    path("register", views.register_request, name="register"),
    path('logout', views.logout_view, name='logout'),
    path("login", views.login_request, name="login"),
]
'''user urls'''