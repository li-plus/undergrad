# Sentence-level Sentiment Classification with RNN

## Extra modification

在参数上加了 `model_name` 参数，用于控制 RNN，LSTM，GRU 三种模型。

添加了 `tensorboard` 绘图的代码，因为我平时用 pytorch 比较多，所以用 `tensorboardX` 会比较顺手些。

## Usage

Install dependencies

```bash
pip install tensorflow-gpu==1.14.0 tensorboardX==1.9
```

Place your data into `codes/data` folder.

Run

```bash
python main.py --model_name=rnn	 	# run RNN
python main.py --model_name=lstm	# run LSTM
python main.py --model_name=gru		# run GRU
```

