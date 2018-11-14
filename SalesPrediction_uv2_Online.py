import tensorflow as tf
import matplotlib as mplt
mplt.use('agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
import csv


class RNNConfig():
    input_size = 1
    num_steps = 5
    lstm_size = 32
    num_layers = 1
    keep_prob = 0.8
    batch_size = 1
    init_learning_rate = 0.01
    learning_rate_decay = 0.99
    test_ratio = 0.2
    fileName = 'store165_2.csv'
    graph = tf.Graph()
    column1_min = 0
    column1_max = 10000
    column1 = 'Sales'
    store = 285
    features = 1



config = RNNConfig()

def segmentation(data):

    seq = [price for tup in data[[config.column1]].values for price in tup]

    seq = np.array(seq)

    # split into number of input_size
    seq = [np.array(seq[i * config.input_size: (i + 1) * config.input_size])
           for i in range(len(seq) // config.input_size)]

    # split into groups of num_steps
    X = np.array([seq[i: i + config.num_steps] for i in range(len(seq) - config.num_steps)])

    y = np.array([seq[i + config.num_steps] for i in range(len(seq) - config.num_steps)])

    return X, y

def scale(data):

    data[config.column1] = (data[config.column1] - config.column1_min) / (config.column1_max - config.column1_min)

    return data

def rescle(test_pred):
    prediction = [(pred * (config.column1_max - config.column1_min)) + config.column1_min for pred in test_pred]

    return prediction


def pre_process():

    store_data = pd.read_csv(config.fileName)

    store_data = store_data.drop(store_data[(store_data.Open == 0) & (store_data.Sales == 0)].index)

    test_len = len(store_data[(store_data.Month == 7) & (store_data.Year == 2015)].index)

    # ---for segmenting original data ---------------------------------
    # train_size = int(len(store_data) * (1.0 - config.test_ratio))
    train_size = int(len(store_data) - test_len)

    #744
    train_data = store_data[:train_size]
    #187
    test_data = store_data[train_size:]
    original_data = store_data[train_size:]

    # -------------- processing train data---------------------------------------

    # log_train_data = convert_log(train_data)
    scaled_train_data = scale(train_data)
    train_X, train_y = segmentation(scaled_train_data)

    # -------------- processing test data---------------------------------------

    # log_train_data = convert_log(test_data)
    scaled_test_data = scale(test_data)
    test_X, test_y = segmentation(scaled_test_data)

    # ----segmenting original test data-----------------------------------------------

    # log_original_data = convert_log(original_data)
    nonescaled_X, nonescaled_y = segmentation(original_data)


    return train_X, train_y, test_X, test_y, nonescaled_y

def RMSPE(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.sqrt(np.mean(np.square(((y_true - y_pred) / y_pred)), axis=0))


def generate_batches(train_X, train_y, batch_size):
    num_batches = int(len(train_X)) // batch_size
    if batch_size * num_batches < len(train_X):
        num_batches += 1

    batch_indices = range(num_batches)
    for j in batch_indices:
        batch_X = train_X[j * batch_size: (j + 1) * batch_size]
        batch_y = train_y[j * batch_size: (j + 1) * batch_size]
        assert set(map(len, batch_X)) == {config.num_steps}
        yield batch_X, batch_y

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def plot( nonescaled_y,prediction_vals,name):

    fig2 = plt.figure()
    fig2 = plt.figure(dpi=100, figsize=(20, 7))
    days = range(len(nonescaled_y))
    plt.plot(days, nonescaled_y, label='true sales')
    plt.plot(days, prediction_vals, label='predicted sales')
    plt.legend(loc='upper left', frameon=False)
    plt.xlabel("day")
    plt.ylabel("sales")
    # plt.ylim((min(test_y), max(test_y)))
    plt.grid(ls='--')
    plt.savefig(name, format='png', bbox_inches='tight', transparent=False)
    plt.close()

def write_results(pred_vals,true_vals,name):
    with open(name, "w") as f:
        writer = csv.writer(f)
        writer.writerows(zip(true_vals, pred_vals))



def train_test():
    train_X, train_y, test_X, test_y, nonescaled_y = pre_process()

    # Add nodes to the graph
    with config.graph.as_default():

        tf.set_random_seed(1)

        learning_rate = tf.placeholder(tf.float32, None, name="learning_rate")
        inputs = tf.placeholder(tf.float32, [None, config.num_steps, config.input_size], name="inputs")
        targets = tf.placeholder(tf.float32, [None, config.input_size], name="targets")
        keep_prob = tf.placeholder(tf.float32, None, name="keep_prob")

        cell = tf.contrib.rnn.LSTMCell(config.lstm_size, state_is_tuple=True, activation=tf.nn.tanh)

        val1, _ = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)

        val = tf.transpose(val1, [1, 0, 2])

        last = tf.gather(val, int(val.get_shape()[0]) - 1, name="last_lstm_output")

        # drop_out = tf.nn.dropout(last, keep_prob)

        # prediction = tf.layers.dense(drop_out, units=1, activation=None)

        # # hidden layer
        # hidden = tf.layers.dense(last, units=20, activation=tf.nn.relu)
        #
        # prediction = tf.contrib.layers.fully_connected(hidden, num_outputs=1, activation_fn=None)
        #

        weight = tf.Variable(tf.truncated_normal([config.lstm_size, config.input_size]))
        bias = tf.Variable(tf.constant(0.1, shape=[config.input_size]))

        prediction = tf.matmul(last, weight) + bias

        loss = tf.losses.mean_squared_error(targets, prediction)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        minimize = optimizer.minimize(loss)

    # --------------------training------------------------------------------------------

    with tf.Session(graph=config.graph) as sess:
        tf.set_random_seed(1)

        tf.global_variables_initializer().run()

        iteration = 1

        for batch_X, batch_y in generate_batches(train_X, train_y, config.batch_size):
            train_data_feed = {
                inputs: batch_X,
                targets: batch_y,
                learning_rate: config.init_learning_rate,
                keep_prob: config.keep_prob
            }

            train_loss, _, value, ya, la = sess.run([loss, minimize, val1, val, last], train_data_feed)

            if iteration % 5 == 0:
                print("Iteration: {}".format(iteration),
                      "Train loss: {:.6f}".format(train_loss))
            iteration += 1

        saver = tf.train.Saver()
        saver.save(sess, "checkpoints_sales/sales_pred.ckpt")

    # --------------------testing------------------------------------------------------

    with tf.Session(graph=config.graph) as sess:
        tf.set_random_seed(1)

        saver.restore(sess, tf.train.latest_checkpoint('checkpoints_sales'))

        test_data_feed = {
            learning_rate: 0.0,
            keep_prob: 1.0,
            inputs: test_X,
            targets: test_y,
        }

        test_pred = sess.run(prediction, test_data_feed)

        pred_vals = rescle(test_pred)

        # pred_vals = convert_from_log(pred_vals)

        pred_vals = np.array(pred_vals)

        pred_vals = pred_vals.flatten()

        nonescaled_y = nonescaled_y.flatten()

        meanSquaredError = mean_squared_error(nonescaled_y, pred_vals)
        rootMeanSquaredError = sqrt(meanSquaredError)
        print("RMSE:", rootMeanSquaredError)
        mae = mean_absolute_error(nonescaled_y, pred_vals)
        print("MAE:", mae)
        mape = mean_absolute_percentage_error(nonescaled_y, pred_vals)
        print("MAPE:", mape)
        rmse_val = RMSPE(nonescaled_y, pred_vals)
        print("RMSPE:", rmse_val)

        write_results( pred_vals,nonescaled_y,"RNN Sales Prediction VS Truth uv Online2 285 results.csv")
        plot(nonescaled_y, pred_vals, "RNN Sales Prediction VS Truth uv Online2 285.png")



if __name__ == '__main__':
    train_test()
