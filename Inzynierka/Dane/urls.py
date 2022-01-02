from django.urls import path
from . import views

urlpatterns = [
    path('', views.home),
    path('general/', views.general),
    path('clubs/', views.clubs),
    path('legia/', views.legia),
    path('manchester/', views.manchester),
    path('roma/', views.roma),
    path('contact/', views.contact),
]
