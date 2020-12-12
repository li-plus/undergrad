import json
import math
import os
import re
import sqlite3
import time

import jieba
from django.shortcuts import render_to_response


def is_space(s):
    return s and s.strip()


def cut_without_space(content):
    seg_list = jieba.cut(content)
    seg_list = [s for s in seg_list if s and s.strip()]
    return seg_list


def home(request):
    context = {}
    if 'q' in request.GET:
        context['is_display'] = True
        context['news_blocks'] = [
            {'title': request.GET['q'], 'author': 'liplus', 'pubtime': 'now',
             'paragraph': 'no comment'},
            {'title': request.GET['q'] + '2', 'author': 'liplus',
             'pubtime': 'a minute ago', 'paragraph': '404 not found'}, ]

    return render_to_response('home.html', context=context)


# receive get request
id2pubtime = {}


def init_id2pubtime():
    global id2pubtime
    if not id2pubtime:
        id2pubtime = get_json('../NewsDataProcess/id2pubtime.json')


def get_json(name):
    with open(name, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def sort_dict_by_value(old_d):
    new_d = sorted(old_d.items(), key=lambda x: x[1], reverse=True)
    return new_d


def search(request):
    conn = sqlite3.connect('../NewsDataProcess/newsweb.db')
    print("Opened database successfully")
    c = conn.cursor()

    start_sec = time.clock()

    init_id2pubtime()

    # init var
    context = {'news_blocks': []}

    # enter empty keyword, return home
    if 'q' not in request.GET:
        return render_to_response('home.html', context={})

    # init keyword
    keyword_str = request.GET['q'].strip()
    if not len(keyword_str):
        return render_to_response('home.html', context={})

    # create selected id
    keyword_list = cut_without_space(keyword_str)

    id_freq = {}

    for keyword in keyword_list:
        cursor = c.execute(
            'SELECT * FROM INVERTED WHERE KEYWORD="{kw}"'.format(kw=keyword))

        for row in cursor:
            news_idx = row[1]
            freq = row[2]
            if news_idx not in id_freq:
                id_freq[news_idx] = freq
            else:
                id_freq[news_idx] += freq

    selected_id = [id for id, _ in sort_dict_by_value(id_freq)]

    # advance 
    start = request.GET.get('from') or str()
    end = request.GET.get('to') or str()
    context['start'] = start
    context['end'] = end

    # advanced
    if start and end:
        selected_id = [sel_idx for sel_idx in selected_id
                       if not (
                        id2pubtime[sel_idx][:10] > end or id2pubtime[sel_idx][
                                                          :10] < start)]

    print(keyword_list)
    # init context
    context['is_display'] = True
    context['keyword_str'] = keyword_str
    context['keyword_highlight'] = ' '.join(keyword_list)
    context['total_result'] = len(selected_id)

    # init pages
    page = 1
    if 'page' in request.GET:
        page = int(request.GET['page'])
    totalpage = math.ceil(len(selected_id) / 10)
    minpage = max(page - 4, 1)
    maxpage = min(minpage + 9, totalpage)
    context['pages'] = []
    for p in range(minpage, maxpage + 1):
        context['pages'].append({'num': p, 'is_current': False})
        if p == page:
            context['pages'][-1]['is_current'] = True

    context['totalpage'] = totalpage

    print(totalpage)
    # create page that shows
    for idx in range(10 * (page - 1), 10 * page):
        if idx >= len(selected_id):
            break
        with open('../NewsDataProcess/news_json/%s.json' % selected_id[idx],
                  'r', encoding='utf-8') as f:
            news_json = json.load(f)
        first_keyword_idx_in_content = news_json['content'].find(keyword)

        if first_keyword_idx_in_content >= 0 and first_keyword_idx_in_content >= 200:
            news_json['content'] = '...' + news_json['content'][
                                           first_keyword_idx_in_content - 100: first_keyword_idx_in_content + 100] + '...'
        else:
            news_json['content'] = news_json['content'][:200] + '...'
        context['news_blocks'].append(
            {'title': news_json['title'], 'pubtime': news_json['pubtime'],
             'content': news_json['content'],
             'url': '/detail/{i}/?q={k}'.format(i=selected_id[idx],
                                                k=keyword_str)})

    context['total_sec'] = (time.clock() - start_sec) / 10

    return render_to_response('search.html', context=context)


# post request
from django.shortcuts import render


def search_post(request):
    ctx = {}
    if request.POST:
        ctx['rlt'] = request.POST['q']
    return render(request, "post.html", ctx)


def allnews(request):
    context = {}
    # init pages
    page = 1
    if 'page' in request.GET:
        page = int(request.GET['page'])
    minpage = max(page - 4, 1)
    maxpage = minpage + 9
    context['pages'] = []
    context['news_blocks'] = []
    context['total_news_num'] = len(os.listdir("../NewsDataProcess/news_json/"))
    for p in range(minpage, maxpage + 1):
        context['pages'].append({'num': p, 'is_current': False})
        if p == page:
            context['pages'][-1]['is_current'] = True

    # create page that shows
    for idx in range(50 * (page - 1), 50 * page):
        if idx >= 30000:
            break
        if not os.path.isfile('../NewsDataProcess/news_json/%s.json' % idx):
            continue

        news_json = get_json('../NewsDataProcess/news_json/%d.json' % idx)
        if not len(news_json):
            print('cannot open file')
            continue
        context['news_blocks'].append(
            {'title': news_json['title'], 'pubtime': news_json['pubtime'],
             'url': '/detail/{i}/'.format(i=idx)})

    return render(request, template_name="allnews.html", context=context)
