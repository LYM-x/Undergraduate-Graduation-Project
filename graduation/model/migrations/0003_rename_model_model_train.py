# Generated by Django 3.2 on 2022-03-14 10:33

from django.conf import settings
from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('model', '0002_model_user'),
    ]

    operations = [
        migrations.RenameModel(
            old_name='model',
            new_name='Model_train',
        ),
    ]
