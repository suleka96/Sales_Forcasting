import tensorflow as tf
import matplotlib as mplt
mplt.use('agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from math import sqrt
from sklearn.linear_model import LinearRegression
from sklearn import linear_model

class RNNConfig():

    input_size = 1
    num_steps = 6
    batch_size = 1
    fileName = 'store165_2.csv'
    graph = tf.Graph()
    features = 4
    column_min_max = [[0, 10000], [1, 7]]
    columns = ['Sales', 'DayOfWeek', 'SchoolHoliday', 'Promo']
    store = 165


config = RNNConfig()

def segmentation(data):

    seq = [price for tup in data[[config.columns[0], config.columns[1], config.columns[2], config.columns[3]]].values for price in tup]

    seq = np.array(seq)

    # split into items of features
    seq = [np.array(seq[i * config.features: (i + 1) * config.features])
           for i in range(len(seq) // config.features)]

    # split into groups of num_steps
    temp_X = np.array([seq[i: i + config.num_steps] for i in range(len(seq) -  config.num_steps)])

    X = []
    for dataslice in temp_X:
        temp = dataslice.flatten()
        X.append(temp)

    X = np.asarray(X)

    y = np.array([seq[i +  config.num_steps] for i in range(len(seq) -  config.num_steps)])

    # get only sales value
    y = [[y[i][0]] for i in range(len(y))]

    y = np.asarray(y)

    return X, y

def scale(data):

    for i in range(len(config.column_min_max)):
        data[config.columns[i]] = (data[config.columns[i]] - config.column_min_max[i][0]) / (
                    (config.column_min_max[i][1]) - (config.column_min_max[i][0]))

    return data


def pre_process():

    store_data = pd.read_csv(config.fileName)

    store_data = store_data.drop(store_data[(store_data.Open == 0) & (store_data.Sales == 0)].index)

    # store_data = store_data.drop(store_data[(store_data.Open != 0) & (store_data.Sales == 0)].index)

    # ---for segmenting original data ---------------------------------
    original_data = store_data.copy()

    test_len = len(store_data[(store_data.Month == 7) & (store_data.Year == 2015)].index)

    train_size = int(len(store_data) - test_len)

    train_data = store_data[:train_size]
    test_data = store_data[train_size:]
    original_data = original_data[train_size:]

    # -------------- processing train data---------------------------------------

    scaled_train_data = scale(train_data)
    train_X, train_y = segmentation(scaled_train_data)

    # -------------- processing test data---------------------------------------

    scaled_test_data = scale(test_data)
    test_X, test_y = segmentation(scaled_test_data)

    # ----segmenting original test data-----------------------------------------------

    nonescaled_X, nonescaled_y = segmentation(original_data)

    return train_X, train_y, test_X, test_y, nonescaled_y


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def RMSPE(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.sqrt(np.mean(np.square(((y_true - y_pred) / y_pred)), axis=0))

def plot(true_vals,pred_vals,name):

    fig2 = plt.figure()
    fig2 = plt.figure(dpi=100, figsize=(20, 7))
    days = range(len(true_vals))
    plt.plot(days, true_vals, label='true sales')
    plt.plot(days, pred_vals, label='predicted sales')
    plt.legend(loc='upper left', frameon=False)
    plt.xlabel("day")
    plt.ylabel("sales")
    # plt.ylim((min(test_y), max(test_y)))
    plt.grid(ls='--')
    plt.savefig(name, format='png', bbox_inches='tight', transparent=False)
    plt.close()

def get_scores(name,pred_vals,nonescaled_y):

    print(name)
    meanSquaredError = mean_squared_error(nonescaled_y, pred_vals)
    rootMeanSquaredError = sqrt(meanSquaredError)
    print("RMSE:", rootMeanSquaredError)
    mae = mean_absolute_error(nonescaled_y, pred_vals)
    print("MAE:", mae)
    mape = mean_absolute_percentage_error(nonescaled_y, pred_vals)
    print("MAPE:", mape)
    rmse_val = RMSPE(nonescaled_y, pred_vals)
    print("RMSPE:", rmse_val)

def rescle(test_pred):

    prediction = [(pred * (config.column_min_max[0][1] - config.column_min_max[0][0])) + config.column_min_max[0][0] for pred in test_pred]

    return prediction

def SGD():

    train_X, train_y, test_X, test_y, nonescaled_y = pre_process()

    clf = linear_model.SGDRegressor()

    for i in range(len(train_X)):
        X, y = train_X[i:i + 1], train_y[i:i + 1]
        clf.partial_fit(X, y)

    predsgdr = clf.predict(test_X)

    pred_vals = rescle(predsgdr)
    pred_vals = np.asarray(pred_vals)

    get_scores("---------SGDRegressor----------", pred_vals, nonescaled_y)

    plot(nonescaled_y, pred_vals, "SGDRegressor Prediction Vs Truth ML Online mv.png")

def PAR():

    train_X, train_y, test_X, test_y, nonescaled_y = pre_process()
    clf = linear_model.PassiveAggressiveRegressor()

    for i in range(len(train_X)):
         X, y = train_X[i:i + 1], train_y[i:i + 1]
         clf.partial_fit(X, y)

    predsgdr = clf.predict(test_X)

    pred_vals = rescle(predsgdr)
    pred_vals = np.asarray(pred_vals)

    get_scores("---------PARegressor----------", pred_vals, nonescaled_y)

    plot(nonescaled_y, pred_vals, "PARegressor Prediction Vs Truth ML Online mv.png")




if __name__ == '__main__':
    SGD()
    PAR()
