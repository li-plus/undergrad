# -*- coding: utf-8 -*-
import json
import os
import pickle

from django.shortcuts import render


# Create your views here.


def hello(request):
    context = {'hello_world': 'Hello World!', 'num_list': [1, 2, 3, 4, 5]}
    return render(request, template_name='hello.html', context=context)


def base(request):
    context = {}
    return render(request, template_name='base.html', context=context)


def derived(request):
    context = {}
    return render(request, template_name="derived.html", context=context)


def tmp(request):
    context = {}
    return render(request, template_name="tmp.html", context=context)


def design1(request):
    context = {}
    return render(request, template_name="design1.html", context=context)


def get_recommendation(idx):
    rec_list = []
    if not os.path.isfile('../NewsDataProcess/recommendation/rec%s.pk' % idx):
        return []
    with open('../NewsDataProcess/recommendation/rec%s.pk' % idx, 'rb') as f:
        rec = pickle.load(f)
    for i, _ in rec:
        with open('../NewsDataProcess/news_json/%s.json' % i, 'r',
                  encoding='utf-8') as f:
            data = json.load(f)
            rec_list.append({'url': '/detail/%s' % i, 'title': data['title'],
                             'pubtime': data['pubtime'],
                             'source': data['source']})
    return rec_list


def detail(request, question_id):
    with open('../NewsDataProcess/news_json/%d.json' % question_id, 'r',
              encoding='utf-8') as f:
        page_dict = json.load(f)
    page_dict['content'] = page_dict['content'].split('\n')
    page_dict['source'] = page_dict['source']
    print(page_dict['source'])
    if 'q' in request.GET:
        keyword = request.GET['q'].strip()
        page_dict['keyword'] = keyword

    rec_list = get_recommendation(question_id)
    page_dict['rec_list'] = rec_list
    print(rec_list)
    print(page_dict.keys())
    return render(request, template_name="detail.html", context=page_dict)
