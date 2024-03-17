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
    path('model1/',views.model1),
    path('model1_back/',views.thread_model1),
    path('model1_down/',views.down),
    path('model2/',views.model2),
    path('model2_back/',views.thread_model2),
    path('model2_down/',views.down),
    path('model3/',views.model3),
    path('model3_back/',views.thread_model3),
    path('model3_down/',views.down),
    path('model4/',views.model4),
    path('model4_back/',views.thread_model4),
    path('model4_down/',views.down),
    path('models/',views.models),
    path('models_back/',views.thread_models),
    path('models_get/',views.models_get),
    path('models_del/',views.models_del),
    path('dels/',views.dels),
    path('models_code_change/',views.models_code_change),
]
urlpatterns+=static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)