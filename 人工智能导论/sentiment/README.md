# Sentiment Classification

## Prerequisites 

- word vector: 下载搜狗新闻词向量，网址为<https://pan.baidu.com/s/1tUghuTno5yOvOx4LXA9-wg>

  将压缩文件解压，得到词向量文件，将其命名为`sgns.sogou.word`，放在resource目录下。

- dataset: 将新浪新闻数据集放在data目录下。

- python packages:

  ```
  numpy==1.16.3
  Keras==2.2.4
  tensorflow-gpu==1.10.0
  scipy==1.0.0
  ```

+ graph visualization tool: graphviz (version 2.38)

工程目录结构如下

<!-- tree sentiment/ -I tree sentiment/ -I "*.pdf|*.pptx|logs|*bz2|__pycache__|*.h5|*.png|*.pk|test.py|*.aux|report.*|model" -->

```
sentiment/
├── data
│   └── sina
│       ├── sinanews.demo
│       ├── sinanews.test
│       └── sinanews.train
├── README.md
├── resource
│   └── sgns.sogou.word
└── src
    ├── models.py
    ├── preprocess.py
    └── utils.py
```

## Pre-Process data

```bash
cd src
python preprocess.py
```

程序会在resource目录下生成`embedding_matrix.pk`和`sina_news.pk`两个文件。

## Train & Evaluate

Text CNN

```bash
python models.py --model text_cnn		# Text CNN 
```

LSTM

```bash
python models.py --model text_lstm		# LSTM
```

Bidirectional LSTM

```bash
python models.py --model text_bi_lstm	# Bidirectional LSTM
```

MLP

```bash
python models.py --model text_mlp		# MLP
```
