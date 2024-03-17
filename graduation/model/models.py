# -*- coding:utf-8 -*-
from django.db import models
import uuid
import os
from user.models import NewUser
# Create your models here.


def user_directory_path1(instance, filename):
    ext = filename.split('.')[-1]
    filename = '{}.{}'.format(uuid.uuid4().hex[:10], ext)
    # return the whole path to the file
    return os.path.join(str(instance.user.id), "model", filename)

def user_directory_path2(instance, filename):
    ext = filename.split('.')[-1]
    filename = '{}.{}'.format(uuid.uuid4().hex[:10], ext)
    # return the whole path to the file
    return os.path.join(str(instance.user.id),'model', "acc", filename)

def user_directory_path3(instance, filename):
    ext = filename.split('.')[-1]
    filename = '{}.{}'.format(uuid.uuid4().hex[:10], ext)
    # return the whole path to the file
    return os.path.join(str(instance.user.id), 'model',"loss", filename)

def user_directory_path4(instance, filename):
    ext = filename.split('.')[-1]
    filename = '{}.{}'.format(uuid.uuid4().hex[:10], ext)
    # return the whole path to the file
    return os.path.join(str(instance.user.id), 'model',"pre", filename)

def user_directory_path5(instance, filename):
    ext = filename.split('.')[-1]
    filename = '{}.{}'.format(uuid.uuid4().hex[:10], ext)
    # return the whole path to the file
    return os.path.join(str(instance.user.id), 'model',"resultdata", filename)

class Model_train(models.Model):
    model_class = models.TextField('模型类别')
    model_name = models.TextField('模型名称')
    model_path = models.FileField('模型路径',upload_to=user_directory_path1)
    loss_pic = models.FileField('loss图',upload_to=user_directory_path3)
    value_index = models.TextField('评估指标')
    result_data = models.FileField('结果数据', upload_to=user_directory_path5,default='')
    data_path = models.TextField('数据路径',default='')
    train_index = models.TextField('训练需要保存的参数',default='')
    user = models.ForeignKey(NewUser, on_delete=models.CASCADE, default='')
    acc_pic = models.FileField('训练准确率图',upload_to=user_directory_path2)
    pre = models.BooleanField('是否预测',default=False)
    pre_result = models.FileField('预测结果',upload_to=user_directory_path4,default='')

