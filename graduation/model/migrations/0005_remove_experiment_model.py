# Generated by Django 3.2 on 2022-03-16 05:46

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('model', '0004_experiment'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='experiment',
            name='model',
        ),
    ]