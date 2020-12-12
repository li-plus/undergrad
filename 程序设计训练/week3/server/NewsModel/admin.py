from NewsModel.models import News, InvertedIndex
from django.contrib import admin

# Register your models here.
admin.site.register(News)
admin.site.register(InvertedIndex)
