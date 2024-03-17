# -*- coding:utf-8 -*-
# @Time : 2022-01-20 16:48
# @Author : 肖紫心
# @File : urls.py
# @Software : PyCharm
from django.contrib import admin
from django.urls import path,include
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    #127.0.0.1:8000/user/login
    path('login/',views.login),
    path('register/',views.register),
    path('setting/',views.setting),
    path('info/',views.user),
    path('superuser/',views.superuser),
    path('change_user/',views.change_user),
    path('del_userno/',views.del_userno),
    path('del_useryes/',views.del_useryes),
    path('add_user/',views.add_user),

]
# urlpatterns+=static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)