# Generated by Django 3.1.1 on 2020-09-15 15:31

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('style_transfer', '0002_auto_20200915_1921'),
    ]

    operations = [
        migrations.AlterField(
            model_name='styles',
            name='Stylefile',
            field=models.ImageField(blank=True, null=True, upload_to='stylefile/'),
        ),
        migrations.AlterField(
            model_name='uploadpics',
            name='Contentfile',
            field=models.ImageField(blank=True, null=True, upload_to='contentupload/', verbose_name=''),
        ),
    ]
