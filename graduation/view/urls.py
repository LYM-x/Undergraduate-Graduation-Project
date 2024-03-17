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
    path('pic2d/',views.IndexView),
    path('pic2d_change/',views.pic2d_change),
    path('pic2d_model/',views.pic2d_model),
    path('pic3d/',views.pic3d),
    path('pic2d_del/',views.pic2d_del),
    path('pic3d_del/',views.pic3d_del),
    path('pic3d_change/',views.pic3d_change),
    path('pic3d_change_map/',views.pic3d_change_map),
    path('pic/',views.pic),
    path('model/',views.model),
    path('file/', views.file),
]
urlpatterns+=static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)