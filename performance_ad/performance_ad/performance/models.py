from django.db import models

# Create your models here.


class Performance(models.Model):
    date = models.DateField()
    channel = models.CharField(max_length=100)
    country = models.CharField(max_length=100)
    os = models.CharField(max_length=100)
    impressions = models.IntegerField()
    clicks = models.IntegerField()
    installs = models.IntegerField()
    spend = models.FloatField()
    revenue = models.FloatField()
