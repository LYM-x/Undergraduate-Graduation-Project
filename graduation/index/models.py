# -*- coding:utf-8 -*-
from django.db import models
from user.models import NewUser
from view.models import Viewpic
import uuid
import os

class Viewleft(models.Model):
    vic_name = models.CharField('图片名称',max_length=50,default='')
    user = models.ForeignKey(NewUser, on_delete=models.CASCADE, default='')
    viewpic = models.CharField('链接图片id',max_length=10,default='-1')

class Viewright(models.Model):
    vic_name = models.CharField('图片名称',max_length=50,default='')
    user = models.ForeignKey(NewUser, on_delete=models.CASCADE, default='')
    viewpic = models.CharField('链接图片id',max_length=10,default='-1')



