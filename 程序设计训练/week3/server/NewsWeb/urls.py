"""HelloWorld URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.conf.urls import url
from django.conf import settings
from django.views import static

from NewsModel import views, search, testdb

urlpatterns = [
    path("admin/", admin.site.urls),
    path('', views.hello),
    path('hello/', views.hello),
    path('base/', views.base),
    path("derived/", views.derived),
    path("tmp/", views.tmp),
    path("design1/", views.design1),
    path("home/", search.home),
    path("search/", search.search),
    path("search-post/", search.search_post),
    path("testdb/", testdb.testdb),
    path("detail/", views.detail),
    path("create-inverted/", testdb.create_inverted_idx),
    path('detail/<int:question_id>/', views.detail, name='detail'),
    path("allnews/", search.allnews),
    url(r'^static/(?P<path>.*)$', static.serve, {'document_root': settings.STATIC_ROOT}, name='static'),
]
