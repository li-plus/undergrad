import json

from NewsModel.models import News, InvertedIndex
from django.http import HttpResponse


def testdb(request):
    test1 = News(name='runoob')
    test1.save()
    return HttpResponse("<p>数据添加成功！</p>")


def create_inverted_idx(request):
    with open("E:/program/NewsDataProcess/inverted_index_content.json", 'r',
              encoding='utf-8') as f:
        data = json.load(f)

    for word in data.keys():
        for piece in data[word]:
            inv_idx = InvertedIndex()
            inv_idx.keyword = word
            inv_idx.freq = piece[1]
            inv_idx.idx = piece[0]
            inv_idx.save()
        print("finish word", word)
    return HttpResponse("<p>数据添加成功！</p>")
