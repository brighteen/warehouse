# Generated by Django 5.1.6 on 2025-02-07 15:23

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("board", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="board",
            name="image",
            field=models.ImageField(blank=True, null=True, upload_to="images/"),
        ),
    ]
