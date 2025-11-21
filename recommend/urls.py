# recommend/urls.py
from django.urls import path
from . import views
from . import api

urlpatterns = [
    path('', views.home, name='home'),
    path('api/recommend/', api.api_recommend, name='api_recommend'),
]
