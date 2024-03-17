# -*- coding:utf-8 -*-
from django.db import models
from user.models import NewUser
import uuid
import os
# Create your models here.
def user_directory_path(instance, filename):
    ext = filename.split('.')[-1]
    filename = '{}.{}'.format(uuid.uuid4().hex[:10], ext)
    # return the whole path to the file
    return os.path.join(str(instance.user.id),"view", '2d',filename)

class Viewpic(models.Model):
    class_name = models.CharField('图片类型',max_length=20,default='')
    pic_name = models.CharField('图片名称',max_length=50,default='')
    data_file = models.FileField('图片数据',upload_to=user_directory_path,default='')
    user = models.ForeignKey(NewUser, on_delete=models.CASCADE, default='')


def user_directory_path3d(instance, filename):
    ext = filename.split('.')[-1]
    filename = '{}.{}'.format(uuid.uuid4().hex[:10], ext)
    # return the whole path to the file
    return os.path.join(str(instance.user.id),"view", '3d',filename)

class Viewpic3d(models.Model):
    class_name = models.CharField('图片类型',max_length=20,default='')
    pic_name = models.CharField('图片名称',max_length=50,default='')
    data_file = models.FileField('图片数据',upload_to=user_directory_path3d,default='')
    other_index = models.TextField('其他信息',default='')
    user = models.ForeignKey(NewUser, on_delete=models.CASCADE, default='')

class Map(models.Model):
    exs_id = models.TextField('预测数据id',default='')
    user = models.ForeignKey(NewUser, on_delete=models.CASCADE, default='')

