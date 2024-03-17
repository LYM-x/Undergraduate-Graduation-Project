# Generated by Django 2.2 on 2022-03-13 02:33

import data.models
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='File',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.TextField(verbose_name='文件名')),
                ('path', models.FileField(upload_to=data.models.user_directory_path, verbose_name='文件路径')),
                ('category', models.CharField(max_length=10, verbose_name='文件类别')),
                ('create_time', models.DateTimeField(auto_now_add=True, verbose_name='创建时间')),
                ('change_time', models.DateTimeField(auto_now=True, verbose_name='更新时间')),
            ],
        ),
    ]