from django.contrib import admin
from django.urls import path
from pred_app import views

urlpatterns = [
    path('',views.index,name='index'),
    path('home',views.home,name='home'),
    path('result',views.result,name='result'),

]
