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
    return os.path.join(str(instance.user.id), "document", filename)

class File(models.Model):
    name = models.TextField('文件名')
    path = models.FileField('文件路径',upload_to=user_directory_path)
    feature_path = models.TextField('特征文件路径',default='')
    category = models.CharField('文件类别',max_length=10)
    create_time = models.DateTimeField('创建时间',auto_now_add=True)
    change_time = models.DateTimeField('更新时间',auto_now=True)
    user = models.ForeignKey(NewUser,on_delete=models.CASCADE,default='')
