# -*- coding:utf-8 -*-
# @Time : 2022-01-20 16:55
# @Author : 肖紫心
# @File : urls.py
# @Software : PyCharm
from django.contrib import admin
from django.urls import path,include
from . import views
from django.conf import settings
from django.conf.urls.static import static
urlpatterns = [
    #127.0.0.1:8000/index
    path('', views.index),
    path('pic/',views.pic)
]
urlpatterns+=static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)