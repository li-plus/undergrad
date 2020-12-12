import argparse
import json

import numpy as np

import viterbi

cfTable = {}
py2ch = {}
lam = 1e-12
mu = 1e-8


def load_py2ch():
    py2ch = {}
    with open('../resource/py2ch.json', encoding='gbk') as f:
        table = f.readlines()
        for line in table:
            line = line.strip().split(' ')
            py2ch[line[0]] = line[1:]
    return py2ch


def load_char_freq():
    with open('../model/charFreqTripleTop.json', encoding='gbk') as f:
        char_freq = json.load(f)
    return char_freq


def freq_sgl(c):
    return cfTable['1'].get(c, 0)


def freq_dbl(s):
    assert len(s) == 2
    return freq_sgl(s[0]) * freq_cond_dbl(s[1], s[0])


def freq_cond_dbl(cur, prev):
    if cur not in cfTable['1'] or prev not in cfTable['1']:
        return 0
    if (prev + cur) in cfTable['2']:  # they occurred together
        return cfTable['2'][prev + cur] / cfTable['1'][prev]
    else:  # they never occurred together
        return lam * cfTable['1'][cur]


def freq_cond_tpl(tri):
    for w in tri:
        if w not in cfTable['1']:
            return 0
    if tri in cfTable['3']:
        return cfTable['3'][tri] / cfTable['2'][tri[:2]]

    return mu * freq_cond_dbl(tri[2], tri[1])


def dbl_model(pinyin):
    # build x
    xs = np.array([freq_sgl(c) for c in py2ch[pinyin[0]]])
    weights = []

    prev = pinyin[0]
    for curr in pinyin[1:]:
        w = np.ndarray((len(py2ch[prev]), len(py2ch[curr])))
        for i1, ch1 in enumerate(py2ch[prev]):
            for i2, ch2 in enumerate(py2ch[curr]):
                w[i1][i2] = freq_cond_dbl(ch2, ch1)

        prev = curr
        weights.append(w)

    path = viterbi.viterbi(xs, weights)

    ans = ''
    for i, p in enumerate(pinyin):
        ans += py2ch[p][path[i]]
    return ans


def get_pairs(p1, p2):
    pairs = []
    for c1 in py2ch[p1]:
        for c2 in py2ch[p2]:
            pairs.append(c1 + c2)
    return pairs


def tpl_model(pinyin):
    for p in pinyin:
        if p not in py2ch:
            return 'wrong spelling!!!!!'
    if len(pinyin) < 3:
        return dbl_model(pinyin)
    weights = []
    ch_pairs = get_pairs(pinyin[0], pinyin[1])
    xs = np.array([freq_dbl(c) for c in ch_pairs])

    win = pinyin[:2]
    for i, p in enumerate(pinyin[2:]):
        win.append(p)  # window of 3 ch
        w = np.ndarray((len(py2ch[win[0]]) * len(py2ch[win[1]]),
                        len(py2ch[win[1]]) * len(py2ch[win[2]])))
        for ip, prev in enumerate(get_pairs(win[0], win[1])):
            for ic, cur in enumerate(get_pairs(win[1], win[2])):
                if not cur[0] == prev[1]:
                    w[ip][ic] = 0
                else:
                    w[ip][ic] = freq_cond_tpl(prev + cur[1])
        weights.append(w)
        win = win[1:]

    path = viterbi.viterbi(xs, weights)

    ans = get_pairs(pinyin[0], pinyin[1])[path[0]][0]
    for i, p in enumerate(path):
        ans += get_pairs(pinyin[i], pinyin[i + 1])[p][1]
    return ans


def main():
    global py2ch
    global cfTable

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='input', type=str, help='input file')
    parser.add_argument('-n', dest='ngram', type=int, default=2, choices=[2, 3],
                        help='n-gram model')
    args = parser.parse_args()

    # load model
    print('loading model')
    py2ch = load_py2ch()
    cfTable = load_char_freq()
    print('model loaded')

    model_map = {2: dbl_model, 3: tpl_model}
    model = model_map[args.ngram]

    if args.input is None:
        print('please type pinyin')
        while True:
            pinyin = input()
            pinyin = pinyin.split()
            if not pinyin:
                break
            ans = model(pinyin)
            print(ans)
    else:
        with open(args.input) as f:
            lines = f.readlines()

        for pinyin in lines:
            pinyin = pinyin.split()
            if not pinyin:
                continue
            ans = tpl_model(pinyin)
            print(ans)


if __name__ == '__main__':
    main()
