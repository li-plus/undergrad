import json
import math
import os
import pickle
import re

import jieba
import numpy as np
from simhash import Simhash

print(Simhash('aa').distance(Simhash('bb')))
print(Simhash('aa').distance(Simhash('aa')))


def get_features(s):
    width = 3
    s = s.lower()
    s = re.sub(r'[^\w]+', '', s)
    return [s[i:i + width] for i in range(max(len(s) - width + 1, 1))]


print(get_features('How are you? I am fine. Thanks.'))

v1 = Simhash(get_features('How are you? I am fine. Thanks.')).value
v2 = Simhash(get_features('How are u? I am fine.     Thanks.')).value
v3 = Simhash(get_features('How r you?I    am fine. Thanks.')).value
v4 = Simhash(get_features('this is a sunny day, have a good time')).value


def hammingDistance(x, y):
    hamming_distance = 0
    s = str(bin(x ^ y))
    for i in range(2, len(s)):
        if int(s[i]) is 1:
            hamming_distance += 1
    return hamming_distance


tokenize = lambda doc: doc.lower().split(" ")

document_0 = "China has a strong economy that is growing at a rapid pace. However politically it differs greatly from the US Economy."
document_1 = "At last, China seems serious about confronting an endemic problem: domestic violence and corruption."
document_2 = "Japan's prime minister, Shinzo Abe, is working towards healing the economic turmoil in his own country for his view on the future of his people."
document_3 = "Vladimir Putin is working hard to fix the economy in Russia as the Ruble has tumbled."
document_4 = "What's the future of Abenomics? We asked Shinzo Abe for his views"
document_5 = "Obama has eased sanctions on Cuba while accelerating those against the Russian Economy, even as the Ruble's value falls almost daily."
document_6 = "Vladimir Putin was found to be riding a horse, again, without a shirt on while hunting deer. Vladimir Putin always seems so serious about things - even riding horses."
document_7 = 'How are you? I am fine. Thanks.'
document_8 = 'How are u? I am fine.     Thanks.'
document_9 = 'How r you?I    am fine. Thanks.'

all_documents = [document_0, document_1, document_2, document_3,
                 document_4, document_5, document_6, document_7, document_8,
                 document_9]

tokenized_documents = [tokenize(d) for d in all_documents]  # tokenized docs
all_tokens_set = set(
    [item for sublist in tokenized_documents for item in sublist])


def jaccard_similarity(query, document):
    intersection = set(query).intersection(set(document))
    union = set(query).union(set(document))
    return len(intersection) / len(union)


def term_frequency(term, tokenized_document):
    return tokenized_document.count(term)


def inverse_document_frequencies(tokenized_documents):
    idf_values = {}
    all_tokens_set = set(
        [item for sublist in tokenized_documents for item in sublist])
    for tkn in all_tokens_set:
        contains_token = map(lambda doc: tkn in doc, tokenized_documents)
        idf_values[tkn] = 1 + \
                          math.log(
                              len(tokenized_documents) / (sum(contains_token)))
    return idf_values


def sublinear_term_frequency(term, tokenized_document):
    return tokenized_document.count(term)


def augmented_term_frequency(term, tokenized_document):
    max_count = max([term_frequency(t, tokenized_document)
                     for t in tokenized_document])
    return (0.5 + (
                (0.5 * term_frequency(term, tokenized_document)) / max_count))


def tfidf(tokenized_documents):
    # tokenized_documents = [tokenize(d) for d in documents]
    idf = inverse_document_frequencies(tokenized_documents)
    tfidf_documents = []
    for document in tokenized_documents:
        doc_tfidf = []
        for term in idf.keys():
            tf = sublinear_term_frequency(term, document)
            doc_tfidf.append(tf * idf[term])
        tfidf_documents.append(doc_tfidf)
    return tfidf_documents


def cut_without_space(content):
    tmp = re.sub(
        r"[\s+\!\/_,()\"\']+|[+——！“”‘’，。？、~@#￥%……&*（）《》；：\s+]+", " ", content)
    # print(tmp)
    seg_list = jieba.cut_for_search(tmp)
    seg_list = [x for x in seg_list if x and x.strip()]
    return seg_list


INTERVAL = 1000

print('init file')
for rnd in range(0, 30):
    all_news_json = {}
    idx_list = []
    MIN_RANGE = rnd * INTERVAL
    MAX_RANGE = (rnd + 1) * INTERVAL
    for i in range(MIN_RANGE, MAX_RANGE):
        fname_i = 'E:\\program\\NewsDataProcess\\news_json\\%s.json' % i
        if not os.path.isfile(fname_i):
            continue

        with open(fname_i, 'r', encoding='utf-8') as f:
            news_json_i = json.load(f)
        if not news_json_i['content']:
            continue
        idx_list.append(i)
        all_news_json[str(i)] = cut_without_space(news_json_i['content'])
        # select the top 20
        no_repeat_word_list = list(set(all_news_json[str(i)]))
        no_repeat_word_list.sort(key=lambda x: all_news_json[str(i)].count(x),
                                 reverse=True)
        no_repeat_word_list = no_repeat_word_list[:40]
        all_news_json[str(i)] = [kw for kw in all_news_json[str(i)] if
                                 kw in no_repeat_word_list]

    print('init recommendation dict')
    rec_dict = {str(k): [] for k in idx_list}

    for idx_list_i in range(len(idx_list)):
        i = idx_list[idx_list_i]
        seg_i = all_news_json[str(i)]

        for idx_list_j in range(idx_list_i + 1, len(idx_list)):
            j = idx_list[idx_list_j]
            seg_j = all_news_json[str(j)]
            tfidf_documents = tfidf([seg_i, seg_j])

            vec = np.array(tfidf_documents, dtype=float).transpose()

            vec = sorted(vec, key=lambda x: x[0] + x[1], reverse=True)[:20]
            vec = np.array(vec, dtype=float).transpose()

            cosdis = vec[0].dot(vec[1]) / math.sqrt(
                vec[0].dot(vec[0]) * vec[1].dot(vec[1]))

            rec_dict[str(i)].append([str(j), cosdis])
            rec_dict[str(j)].append([str(i), cosdis])

        rec_dict[str(i)].sort(key=lambda x: x[1], reverse=True)
        rec_dict[str(i)] = rec_dict[str(i)][:5]
        print('finish' + str(i))
        print(rec_dict[str(i)])
        with open('E:\\program\\NewsDataProcess\\recommendation\\rec%d.pk' % i,
                  'wb') as f:
            pickle.dump(rec_dict[str(i)], f)
