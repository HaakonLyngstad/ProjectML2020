from django.contrib import admin
from django.urls import path, include
from app import views
from django.contrib import admin

urlpatterns = [
    path('', views.main_view, name='homepage'),
    path('predicate', views.predicate, name='predicate'),
    path('support', views.support, name='support'),
]
