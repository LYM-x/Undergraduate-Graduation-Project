# Generated by Django 3.2 on 2022-05-04 13:26

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('index', '0002_alter_viewindex_viewpic'),
    ]

    operations = [
        migrations.AlterField(
            model_name='viewindex',
            name='viewpic',
            field=models.CharField(default='-1', max_length=10, verbose_name='链接图片id'),
        ),
    ]
