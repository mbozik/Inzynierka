from django.db import models

# Create your models here.
class Piłkarze(models.Model):
    name = models.CharField(max_length=200)
    wage = models.CharField(max_length=200)
    value = models.IntegerField(max_length=200)
    position = models.CharField(max_length=200)
    overall = models.IntegerField(max_length=200)
    age = models.IntegerField(max_length=200)
