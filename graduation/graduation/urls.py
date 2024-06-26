"""graduation URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path,include
from . import views
from django.conf import settings
from django.contrib.staticfiles.views import serve
from django.urls import re_path
from django.views.static import serve as static_serve

def return_static(request, path, insecure=True, **kwargs):
  return serve(request, path, insecure, **kwargs)

urlpatterns = [
    path('', views.index),
    #127.0.0.1:8000/index
    path('index/', include('index.urls')),
    #127.0.0.1:8000/admin
    path('admin/', admin.site.urls),
    #127.0.0.1:8000/user
    path('user/',include('user.urls')),
    path('model/', include('model.urls')),
    path('data/', include('data.urls')),
    path('view/', include('view.urls')),
    re_path(r'^static/(?P<path>.*)$', return_static, name='static'),
    re_path(r'^media/(?P<path>.*)$' , static_serve, { 'document_root' : settings.MEDIA_ROOT}),
]
