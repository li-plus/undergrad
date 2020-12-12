import argparse
import os

import keras
import numpy as np
import utils
from keras.layers import (
    Dense, Dropout, Flatten, Conv1D, MaxPooling1D, LSTM, concatenate, Embedding,
    Bidirectional, TimeDistributed)
from keras.losses import categorical_crossentropy
from keras.models import load_model, Model
from keras.optimizers import Adam
from keras.utils import plot_model
from scipy.stats import pearsonr
from sklearn.metrics import f1_score

seq_length = 600
model_path = '../model/'


def construct_text_cnn(seq_length, num_classes, embedding_matrix):
    ngrams = [1, 2, 3, 4, 5]
    convs = []
    input_seq = keras.Input(shape=(seq_length,), name='input')
    embed = Embedding(input_dim=embedding_matrix.shape[0],
                      output_dim=embedding_matrix.shape[1],
                      weights=[embedding_matrix], input_length=seq_length,
                      trainable=False, name='embed')(input_seq)

    for n in ngrams:
        conv = Conv1D(128, n, activation='relu', padding='valid',
                      name='conv%d' % n)(embed)
        conv = MaxPooling1D(seq_length - n + 1, strides=1,
                            name='maxpool%d' % n)(conv)
        convs.append(conv)

    x = concatenate(convs, axis=1, name='concat')
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5, name='dropout')(x)
    output = Dense(num_classes, activation='softmax', name='output')(x)

    model = Model(input_seq, output)
    model.compile(loss=categorical_crossentropy, optimizer=Adam(),
                  metrics=['accuracy'])
    return model


def construct_text_lstm(seq_length, num_classes, embedding_matrix):
    input_seq = keras.Input(shape=(seq_length,), name='input')
    embed = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                      weights=[embedding_matrix], input_length=seq_length,
                      trainable=False, name='embed')(input_seq)

    x = LSTM(64, name='lstm')(embed)
    x = Dropout(0.5, name='dropout')(x)
    output = Dense(num_classes, activation='softmax', name='output')(x)

    model = Model(input_seq, output)
    model.compile(loss=categorical_crossentropy, optimizer=Adam(),
                  metrics=['accuracy'])
    return model


def construct_text_bi_lstm(seq_length, num_classes, embedding_matrix):
    input_seq = keras.Input(shape=(seq_length,), name='input')
    embed = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                      weights=[embedding_matrix], input_length=seq_length,
                      trainable=False, name='embed')(input_seq)

    x = Bidirectional(LSTM(128, return_sequences=True, name='lstm'),
                      name='bidirectional')(embed)
    x = TimeDistributed(Dense(64, activation='relu', name='fc'),
                        name='time_distributed')(x)
    x = MaxPooling1D(seq_length, name='maxpool')(x)
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5, name='dropout')(x)
    output = Dense(num_classes, activation='softmax', name='output')(x)

    model = Model(input_seq, output)
    model.compile(loss=categorical_crossentropy, optimizer=Adam(),
                  metrics=['accuracy'])
    return model


def construct_text_mlp(seq_len, num_classes, embedding_matrix):
    input_seq = keras.Input(shape=(seq_len,), name='input')
    embed = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                      weights=[embedding_matrix],
                      trainable=False, name='embed')(input_seq)

    x = Flatten(name='flatten')(embed)
    x = Dense(128, activation='relu', name='fc1')(x)
    x = Dropout(0.5, name='dropout')(x)

    output = Dense(num_classes, activation='softmax', name='output')(x)

    model = Model(input_seq, output)
    model.compile(loss=categorical_crossentropy, optimizer=Adam(),
                  metrics=['accuracy'])
    return model


def construct_model(model_name, seq_length, num_classes, embedding_matrix):
    if model_name == 'text_cnn':
        return construct_text_cnn(seq_length, num_classes, embedding_matrix)
    if model_name == 'text_lstm':
        return construct_text_lstm(seq_length, num_classes, embedding_matrix)
    if model_name == 'text_bi_lstm':
        return construct_text_bi_lstm(seq_length, num_classes, embedding_matrix)
    if model_name == 'text_mlp':
        return construct_text_mlp(seq_length, num_classes, embedding_matrix)

    raise Exception("invalid model name")


def train(model_name):
    embedding_matrix = utils.load_embedding_matrix()

    model = construct_model(model_name, seq_length, 8, embedding_matrix)

    model.summary()
    plot_model(model, to_file=model_path + model_name + '.png',
               show_shapes=False)

    x_train, y_train, x_test, y_test, label_names = utils.load_sina_news()

    tb = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0,
                                     write_graph=True, write_images=True)
    ckpt = keras.callbacks.ModelCheckpoint(
        model_path + model_name + '.h5', monitor='val_loss', mode='min',
        verbose=1, save_best_only=True, period=1)

    history = model.fit(x_train, y_train, batch_size=256, epochs=100,
                        validation_split=0.16, callbacks=[tb, ckpt], verbose=2)


def evaluate(model_name):
    model = load_model(model_path + model_name + ".h5")
    x_train, y_train, x_test, y_test, label_names = utils.load_sina_news()

    y_pred = model.predict(x_test)

    y_test_new = np.zeros(y_test.shape, dtype="int")
    for yp, yt, yn in zip(y_pred, y_test, y_test_new):
        if yt[np.argmax(yp)] == 1:  # predicted one of the max labels
            yn[np.argmax(yp)] = 1
        else:  # false prediction
            yn[np.argmax(yt)] = 1
    y_test = y_test_new

    # calculate correlation coefficient
    coefs = []
    for pred, label in zip(y_pred, y_test):
        coef, _ = pearsonr(pred, label)
        coefs.append(coef)
    coef = np.average(coefs)

    # accuracy
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=-1)
    acc = sum(y_pred == y_test) / 2228

    # f1 score
    f1 = f1_score(y_test, y_pred, average='macro')

    print("testing", model_name)
    print('acc', acc)
    print('f1-score', f1)
    print('coef', coef)


def main():
    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='name of model')
    params = parser.parse_args()

    model_name = params.model
    train(model_name)
    evaluate(model_name)


if __name__ == "__main__":
    main()
