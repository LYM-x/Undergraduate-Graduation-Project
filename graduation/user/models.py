# -*- coding:utf-8 -*-
from django.db import models
from django.contrib.auth.models import AbstractUser
import uuid
import os
# Create your models here.

# class Last_url(models.Model):
#     last_url = models.CharField('上一次路由地址',max_length=40,default='')
#     create_time = models.DateTimeField('创建时间',auto_now_add=True)
#     user = models.ForeignKey(User,on_delete=models.CASCADE)

def user_directory_path(instance, filename):
    ext = filename.split('.')[-1]
    filename = '{}.{}'.format(uuid.uuid4().hex[:10], ext)
    # return the whole path to the file
    return os.path.join(str(instance.id), "picture", filename)

class NewUser(AbstractUser):
    info = models.TextField('简介')
    img = models.FileField('用户头像',upload_to=user_directory_path,default='')