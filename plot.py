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
from math import sin
from math import radians
from matplotlib import pyplot
np.random.seed(1)
tf.set_random_seed(1)

class Config():
    input_size = 1
    num_steps = 7#5
    lstm_size = 32
    num_layers = 1
    keep_prob = 0.8
    batch_size = 8
    init_learning_rate = 0.01
    learning_rate_decay = 0.99
    init_epoch = 5  # 5
    max_epoch = 60  # 100 or 50
    test_ratio = 0.2
    fileName = 'store165_2.csv'
    graph = tf.Graph()
    features = 4
    column_min_max = [[0,11000], [1,7]]
    columns = ['Sales', 'DayOfWeek','SchoolHoliday', 'Promo']

config = Config()

# create a differenced series
def difference(dataset, interval=1):

    return dataset



# invert differenced forecast
def inverse_difference(last_ob, value):
	return value + last_ob




#
# def plot_sales():
#
#     # store_data = pd.read_csv(config.fileName)
#
#     # store_data = store_data.drop(store_data[(store_data.Open == 0) & (store_data.Sales == 0)].index)
#
#     # store_data = store_data[(store_data.Month == 2) & (store_data.Year == 2013)]
#
#     fig = plt.figure()
#     fig = plt.figure(dpi=100, figsize=(20, 7))
#     days = range(len(store_data))
#     plt.plot(days, store_data[config.columns[0]], label='pred sales')
#     plt.legend(loc='upper left', frameon=False)
#     plt.xlabel("day")
#     plt.ylabel("sales")
#     plt.grid(ls='--')
#     plt.savefig("sales graph.png", format='png', bbox_inches='tight', transparent=False)
#     plt.close()

if __name__ == '__main__':
    store_data = pd.read_csv(config.fileName)
    # store_data = store_data.drop(store_data[(store_data.Open == 0) & (store_data.Sales == 0)].index)
    store_data = store_data[(store_data.Month == 1) & (store_data.Year == 2013)]

    fig = plt.figure()
    fig = plt.figure(dpi=100, figsize=(20, 7))
    days = range(len(store_data))
    plt.plot(days, store_data[config.columns[0]], label='pred sales')
    plt.legend(loc='upper left', frameon=False)
    plt.xlabel("day")
    plt.ylabel("sales")
    plt.grid(ls='--')
    plt.savefig("sales graph.png", format='png', bbox_inches='tight', transparent=False)
    plt.close()

    # difference the dataset
    store_data['Sales'] = store_data['Sales'].diff(periods=14)

    fig = plt.figure()
    fig = plt.figure(dpi=100, figsize=(20, 7))
    days = range(len(store_data))
    plt.plot(days, store_data.Sales, label='pred sales')
    plt.legend(loc='upper left', frameon=False)
    plt.xlabel("day")
    plt.ylabel("sales")
    plt.grid(ls='--')
    plt.savefig("sales graph 2.png", format='png', bbox_inches='tight', transparent=False)
    plt.close()

    # pyplot.plot(diff)
    # pyplot.show()
    # invert the difference
    # inverted = [inverse_difference(store_data[i], diff[i]) for i in range(len(diff))]
    # pyplot.plot(inverted)
    # pyplot.show()