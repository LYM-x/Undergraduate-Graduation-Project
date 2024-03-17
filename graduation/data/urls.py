# -*- coding:utf-8 -*-
# @Time : 2022-01-20 16:59
# @Author : 肖紫心
# @File : urls.py
# @Software : PyCharm
from django.urls import path,include
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('spider/',views.spider),
    path('manage/',views.manage),
    path('look_file/', views.look_file),
    path('del/', views.delete),
    path('change_del/', views.change_del),
    path('change_add/', views.change_add),
    path('down/', views.down),
    path('data_get/', views.data_get),
]
urlpatterns+=static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)