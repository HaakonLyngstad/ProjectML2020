from django.urls import path
from app import views

urlpatterns = [
    path('', views.main_view, name='homepage'),
    path('methods', views.predicate, name='methods'),
    path('support', views.support, name='support'),
    path('api/predict', views.api_predicate, name='api'),
]
