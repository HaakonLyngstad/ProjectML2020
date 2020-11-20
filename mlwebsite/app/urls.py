from django.contrib import admin
from django.urls import path, include
from app import views
from django.contrib import admin

urlpatterns = [
    path('', views.main_view, name='homepage'),
    path('methods', views.methods, name='methods'),
    path('support', views.support, name='support'),
    path('bagging', views.bagging_view, name='bagging'),
    path('boosting', views.boosting_view, name='boosting'),
    path('stacking', views.stacking_view, name='stacking'),
]
