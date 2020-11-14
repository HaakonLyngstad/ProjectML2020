from django.db import models


# Create your models here.
class Dataset(models.Model):
    title = models.CharField(max_length=200)
    comment = models.CharField(max_length=1000)
    data_file = models.FileField(upload_to='datasets')
