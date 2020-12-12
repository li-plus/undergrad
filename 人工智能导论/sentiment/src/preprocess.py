import re

import numpy as np
import utils
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

seq_length = 600


def extract_chinese(text):
    return re.sub(r'[^\u4e00-\u9fa5]+', ' ', text).strip()


def news_to_text(fname, is_test):
    word_size = 0
    num_classes = 8
    with open(fname, 'r', encoding='utf-8') as f:
        data = f.read()

    data = data.strip().split('\n')
    news_size = len(data)

    print("total news size", news_size)

    ys = np.zeros((news_size, num_classes), dtype='int8')
    xs = []
    for news_id, d in enumerate(data):
        result = re.findall(
            r'.+感动:(\d+) 同情:(\d+) 无聊:(\d+) 愤怒:(\d+) 搞笑:(\d+) 难过:(\d+) 新奇:(\d+) 温馨:(\d+)\t(.*)',
            d)

        if len(result) != 1:
            print('pre process error')
            exit(0)

        labels = [int(x) for x in result[0][:8]]
        text = result[0][8]
        if is_test:
            max_comment = max(labels)
            ys[news_id] = [1 if c == max_comment else 0 for c in labels]
        else:
            max_arg = np.argmax(labels)
            ys[news_id, max_arg] = 1

        text = extract_chinese(text)

        word_size += len(text)
        xs.append(text)

    return xs, ys


def to_embedding_matrix(tk):
    print('loading vectors')
    vectors, iw, wi, dim = utils.read_vectors(0)
    mat = np.zeros((len(tk.word_index) + 1, dim))

    for w, i in tk.word_index.items():
        if w in wi:
            mat[i] = vectors[w]
        else:
            mat[i] = np.zeros((dim,))
    return mat


def main():
    label_names = ['感动', '同情', '无聊', '愤怒', '搞笑', '难过', '新奇', '温馨']

    print('processing')
    x_train, y_train = news_to_text('../data/sina/sinanews.train', False)
    x_test, y_test = news_to_text('../data/sina/sinanews.test', True)

    corpus = x_train + x_test
    tk = Tokenizer(num_words=50000)
    tk.fit_on_texts(corpus)

    x_train = tk.texts_to_sequences(x_train)
    x_train = pad_sequences(x_train, maxlen=seq_length, truncating='post',
                            padding='post')

    x_test = tk.texts_to_sequences(x_test)
    x_test = pad_sequences(x_test, maxlen=seq_length)

    mat = to_embedding_matrix(tk)

    print('dumping')
    utils.dump_sina_news(x_train, y_train, x_test, y_test, label_names)
    utils.dump_embedding_matrix(mat)


if __name__ == "__main__":
    main()
