# -*- coding:utf-8 -*-
# @Time : 2022-01-20 16:04
# @Author : 肖紫心
# @File : views.py
# @Software : PyCharm
from django.shortcuts import render
from django.http import HttpResponse,HttpResponseRedirect

def index(request):
    return HttpResponseRedirect('/index')
