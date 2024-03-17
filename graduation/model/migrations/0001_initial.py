# Generated by Django 2.2 on 2022-03-13 02:33

from django.db import migrations, models
import model.models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='model',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('model_class', models.TextField(verbose_name='模型类别')),
                ('model_name', models.TextField(verbose_name='模型名称')),
                ('model_path', models.FileField(upload_to=model.models.user_directory_path1, verbose_name='模型路径')),
                ('loss_pic', models.FileField(upload_to=model.models.user_directory_path3, verbose_name='loss图')),
                ('value_index', models.TextField(verbose_name='评估指标')),
                ('acc_pic', models.FileField(upload_to=model.models.user_directory_path2, verbose_name='训练准确率图')),
            ],
        ),
    ]
