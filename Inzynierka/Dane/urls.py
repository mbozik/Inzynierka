from django.urls import path
from . import views

urlpatterns = [
    path('', views.home),
    path('general/', views.general),
    path('clubs/', views.clubs),
    path('dashboard/', views.dashboard),
    path('contact/', views.contact),
    path('table/', views.table, name ="table"),
]
