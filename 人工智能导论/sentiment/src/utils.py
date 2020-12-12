import pickle

import numpy as np

path_vectors = '../resource/sgns.sogou.word'
path_embedding_matrix = '../resource/embedding_matrix.pk'
path_sina_news = '../resource/sina_news.pk'


def vectors_to_embedding_matrix(vectors, iw, wi, dim):
    embedding_matrix = np.empty((len(iw), dim))
    for k, v in vectors.items():
        embedding_matrix[wi[k]] = v
    embedding_matrix[0] = 0
    return embedding_matrix


def dump_embedding_matrix(mat):
    with open(path_embedding_matrix, 'wb') as f:
        pickle.dump(mat, f)


def load_embedding_matrix():
    with open(path_embedding_matrix, 'rb') as f:
        embedding_matrix = pickle.load(f)
    return embedding_matrix


def read_vectors(topn):  # read top n word vectors, i.e. top is 10000
    lines_num, dim = 0, 0
    vectors = {}  # word vector
    iw = []  # index to word
    wi = {}  # word to index
    with open(path_vectors, encoding='utf-8', errors='ignore') as f:
        first_line = True
        for line in f:
            if first_line:
                first_line = False
                dim = int(line.rstrip().split()[1])
                continue
            lines_num += 1
            tokens = line.rstrip().split(' ')
            vectors[tokens[0]] = np.asarray([float(x) for x in tokens[1:]])
            iw.append(tokens[0])
            if topn != 0 and lines_num >= topn:
                break
    for i, w in enumerate(iw):
        wi[w] = i
    return vectors, iw, wi, dim


def load_sina_news():
    with open(path_sina_news, 'rb') as f:
        data = pickle.load(f)

    x_train, y_train, x_test, y_test = data['x_train'], data['y_train'], data[
        'x_test'], data['y_test']
    label_names = data['label_names']

    return x_train, y_train, x_test, y_test, label_names


def dump_sina_news(x_train, y_train, x_test, y_test, label_names):
    data = {'x_train': x_train, 'y_train': y_train, 'x_test': x_test,
            'y_test': y_test, 'label_names': label_names}
    with open(path_sina_news, 'wb') as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    pass
