import tensorflow as tf
import matplotlib as mplt
mplt.use('agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
import csv
np.random.seed(1)
tf.set_random_seed(1)


class RNNConfig():
    input_size = 1
    num_steps = 6#5
    lstm_size = 32
    num_layers = 1
    keep_prob = 0.8
    batch_size = 8
    init_learning_rate = 0.01
    learning_rate_decay = 0.99
    init_epoch = 5  # 5
    max_epoch = 60  # 100 or 50
    test_ratio = 0.2
    fileName = 'store165_lagged2.csv'
    graph = tf.Graph()
    features = 4
    column_min_max = [[0,10000], [1,7]]
    columns = ['Sales', 'DayOfWeek','SchoolHoliday', 'Promo']

config = RNNConfig()

def segmentation(data):

    seq = [price for tup in data[config.columns].values for price in tup]

    seq = np.array(seq)

    # split into items of features
    seq = [np.array(seq[i * config.features: (i + 1) * config.features])
           for i in range(len(seq) // config.features)]

    # split into groups of num_steps
    X = np.array([seq[i: i + config.num_steps] for i in range(len(seq) -  config.num_steps)])

    y = np.array([seq[i +  config.num_steps] for i in range(len(seq) -  config.num_steps)])

    # get only sales value
    y = [[y[i][0]] for i in range(len(y))]

    y = np.asarray(y)

    return X, y

def scale(data):

    for i in range (len(config.column_min_max)):
        data[config.columns[i]] = (data[config.columns[i]] - config.column_min_max[i][0]) / ((config.column_min_max[i][1]) - (config.column_min_max[i][0]))

    return data

def rescle(test_pred):

    prediction = [(pred * (config.column_min_max[0][1] - config.column_min_max[0][0])) + config.column_min_max[0][0] for pred in test_pred]

    return prediction


def pre_process():
    store_data = pd.read_csv(config.fileName,skipfooter=1)

    store_data = store_data.drop(store_data[(store_data.Open == 0) & (store_data.Sales == 0)].index)
    #
    # store_data = store_data.drop(store_data[(store_data.Open != 0) & (store_data.Sales == 0)].index)

    # ---for segmenting original data --------------------------------
    original_data = store_data.copy()

    # train_size = int(len(store_data) * (1.0 - config.test_ratio))

    test_len = len(store_data[(store_data.Month == 7) & (store_data.Year == 2015)].index)


    train_size = int(len(store_data) - test_len)

    train_data = store_data[:train_size]
    test_data = store_data[(train_size-config.num_steps):]
    original_data = store_data[(train_size-config.num_steps):]

    # -------------- processing train data---------------------------------------
    scaled_train_data = scale(train_data)
    train_X, train_y = segmentation(scaled_train_data)


    # -------------- processing test data---------------------------------------
    scaled_test_data = scale(test_data)
    test_X, test_y = segmentation(scaled_test_data)


    # ----segmenting original test data-----------------------------------------------
    nonescaled_X, nonescaled_y = segmentation(original_data)



    return train_X, train_y, test_X, test_y, nonescaled_y


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

def RMSPE(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.sqrt(np.mean(np.square(((y_true - y_pred) / y_pred)), axis=0))

def plot(true_vals,pred_vals,name):
    fig = plt.figure()
    fig = plt.figure(dpi=100, figsize=(20, 7))
    days = range(len(true_vals))
    plt.plot(days, pred_vals, label='pred sales')
    plt.plot(days, true_vals, label='truth sales')
    plt.legend(loc='upper left', frameon=False)
    plt.xlabel("day")
    plt.ylabel("sales")
    plt.grid(ls='--')
    plt.savefig(name, format='png', bbox_inches='tight', transparent=False)
    plt.close()

def write_results(true_vals,pred_vals,name):

    with open(name, "w") as f:
        writer = csv.writer(f)
        writer.writerows(zip(true_vals, pred_vals))


def train_test():
    train_X, train_y, test_X, test_y, nonescaled_y = pre_process()


    # Add nodes to the graph
    with config.graph.as_default():

        tf.set_random_seed(1)

        learning_rate = tf.placeholder(tf.float32, None, name="learning_rate")
        inputs = tf.placeholder(tf.float32, [None, config.num_steps, config.features], name="inputs")
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

        loss = tf.losses.mean_squared_error(targets,prediction)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        minimize = optimizer.minimize(loss)


    # --------------------training------------------------------------------------------

    with tf.Session(graph=config.graph) as sess:
        tf.set_random_seed(1)

        tf.global_variables_initializer().run()

        iteration = 1

        learning_rates_to_use = [
            config.init_learning_rate * (
                    config.learning_rate_decay ** max(float(i + 1 - config.init_epoch), 0.0)
            ) for i in range(config.max_epoch)]

        for epoch_step in range(config.max_epoch):

            current_lr = learning_rates_to_use[epoch_step]

            for batch_X, batch_y in generate_batches(train_X, train_y, config.batch_size):
                train_data_feed = {
                    inputs: batch_X,
                    targets: batch_y,
                    learning_rate: current_lr,
                    keep_prob: config.keep_prob
                }

                train_loss, _, value = sess.run([loss, minimize, val1], train_data_feed)

                if iteration % 5 == 0:
                    print("Epoch: {}/{}".format(epoch_step, config.max_epoch),
                          "Iteration: {}".format(iteration),
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

        pred_vals = np.array(pred_vals)

        pred_vals = pred_vals.flatten()

        pred_vals = pred_vals.tolist()

        nonescaled_y = nonescaled_y.flatten()

        nonescaled_y = nonescaled_y.tolist()

        plot(nonescaled_y, pred_vals, "Sales Prediction VS Truth mv lagged 2.png")
        write_results(nonescaled_y, pred_vals, "Sales Prediction batch mv lagged 2 results.csv")

        meanSquaredError = mean_squared_error(nonescaled_y, pred_vals)
        rootMeanSquaredError = sqrt(meanSquaredError)
        print("RMSE:", rootMeanSquaredError)
        mae = mean_absolute_error(nonescaled_y, pred_vals)
        print("MAE:", mae)
        mape = mean_absolute_percentage_error(nonescaled_y, pred_vals)
        print("MAPE:", mape)
        rmse_val = RMSPE(nonescaled_y, pred_vals)
        print("RMSPE:", rmse_val)


if __name__ == '__main__':
    train_test()