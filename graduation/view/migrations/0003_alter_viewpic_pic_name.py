# Generated by Django 3.2 on 2022-04-29 02:10

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('view', '0002_viewpic3d_other_index'),
    ]

    operations = [
        migrations.AlterField(
            model_name='viewpic',
            name='pic_name',
            field=models.CharField(default='', max_length=50, verbose_name='图片名称'),
        ),
    ]