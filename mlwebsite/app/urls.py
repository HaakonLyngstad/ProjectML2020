from django.urls import path
from app import views

urlpatterns = [
    path('', views.main_view, name='homepage'),
    path('methods', views.methods, name='methods'),
    path('support', views.support, name='support'),
    path('api/predict', views.api_predicate, name='api'),
    path('bagging', views.bagging_view, name='bagging'),
    path('boosting', views.boosting_view, name='boosting'),
    path('stacking', views.stacking_view, name='stacking'),
]
