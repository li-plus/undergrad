# Artificial Neural Network Homework 1

## Getting Started

A virtual environment is recommended.

```bash
conda create --name mnist python=3.6
conda activate mnist
```

Install requirements.

```bash
pip install -r requirements.txt
```

Prepare MNIST dataset in directory `data`. The project structure can be referred as follows.

```
codes
├── data
│   ├── t10k-images-idx3-ubyte
│   ├── t10k-labels-idx1-ubyte
│   ├── train-images-idx3-ubyte
│   └── train-labels-idx1-ubyte
├── layers.py
├── load_data.py
├── loss.py
├── network.py
├── README.md
├── requirements.txt
├── run_mlp.py
├── solve_net.py
└── utils.py
```

The main entrance is `run_mlp.py`. For example, to run MLP with 2 layers with Relu activation and Softmax cross entropy loss, execute the following command.

```bash
python run_mlp.py --layers 2 --activation relu --loss sce
```

For more options, run

```bash
python run_mlp.py --help
```

## Extra modification

+ Modules in `layers.py` and `loss.py` are implemented. 
+ Training loss and accuracy of each iteration is saved with both `tensorboardx` and `pickle` for future analysis.
+ Training time is computed and printed out.
+ Hyper parameters can be adjusted via command line arguments. 

