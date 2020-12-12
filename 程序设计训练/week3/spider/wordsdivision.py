import json
import os
import re

import jieba

DATA_PATH = os.path.join(os.getcwd(), 'news_json')


def cut_without_space(content):
    tmp = re.sub(
        r"[\s+\!\/_,()\"\']+|[+——！“”‘’，。？、~@#￥%……&*（）《》；：\s+]+", " ", content)
    # print(tmp)
    seg_list = jieba.cut_for_search(tmp)
    seg_list = filter(is_space, seg_list)
    return seg_list


def is_space(s):
    return s and s.strip()


def sort_dict_by_value(old_d):
    new_d = sorted(old_d.items(), key=lambda x: x[1], reverse=True)
    return new_d


def create_inverted_index():
    word2id = {}
    for idx in range(0, 30000):
        if not os.path.isfile(os.path.join(DATA_PATH, str(idx) + '.json')):
            print(str(idx) + '.json: file not found')
            continue
        with open(os.path.join(DATA_PATH, str(idx) + '.json'), 'r',
                  encoding='utf-8') as f:
            news_dict = json.load(f)
        content = news_dict['content']
        title = news_dict['title']

        content = re.sub(
            r"[\s+\!\/_,()\"\']+|[+——！“”‘’，。？、~@#￥%……&*（）《》；：\s+]+", " ",
            content)
        title = re.sub(
            r"[\s+\!\/_,()\"\']+|[+——！“”‘’，。？、~@#￥%……&*（）《》；：\s+]+", " ", title)

        seg_list_content = jieba.cut_for_search(content)
        seg_list_content = filter(is_space, seg_list_content)

        seg_list_title = jieba.cut_for_search(title)
        seg_list_title = filter(is_space, seg_list_title)

        for word in seg_list_content:
            # process word2id
            if word not in word2id.keys():
                word2id[word] = {str(idx): 1}
            else:
                if str(idx) not in word2id[word].keys():
                    word2id[word].update({str(idx): 1})
                else:
                    word2id[word][str(idx)] += 1

        for word in seg_list_title:
            # process word2id
            if word not in word2id.keys():
                word2id[word] = {str(idx): 1}
            else:
                if str(idx) not in word2id[word].keys():
                    word2id[word].update({str(idx): 1})
                else:
                    word2id[word][str(idx)] += 20

    # sort and save
    for key in word2id.keys():
        word2id[key] = sort_dict_by_value(word2id[key])

    return word2id


def create_id_pubtime():
    id2pubtime = {}
    for idx in range(0, 30000):
        if not os.path.isfile(os.path.join(DATA_PATH, str(idx) + '.json')):
            print(str(idx) + '.json: file not found')
            continue
        with open(os.path.join(DATA_PATH, str(idx) + '.json'), 'r',
                  encoding='utf-8') as f:
            news_dict = json.load(f)

        res = re.findall(r'(.+)\s+.+', news_dict['pubtime'])
        if not res:
            print("cannot pase data")
        pubtime = res[0]
        id2pubtime.update({str(idx): pubtime})

    with open('id2pubtime.json', 'w', encoding='utf-8') as f:
        json.dump(id2pubtime, f)


if __name__ == "__main__":
    word2id = create_inverted_index()
    with open('inverted_index.json', 'w', encoding='utf-8') as f:
        json.dump(word2id, f, ensure_ascii=False)
