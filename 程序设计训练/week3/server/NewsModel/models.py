from django.db import models


# Create your models here.


class News(models.Model):
    name = models.CharField(max_length=20)


class SingleNews(models.Model):
    content = models.TextField()
    title = models.CharField(max_length=100)
    pubtime = models.CharField(max_length=50)
    source = models.CharField(max_length=30)


class InvertedIndex(models.Model):
    keyword = models.CharField(max_length=50)
    freq = models.IntegerField()
    idx = models.CharField(max_length=10)
