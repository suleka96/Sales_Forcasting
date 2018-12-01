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
        data[config.columns[i]] = (data[config.columns[i]] - config.column_min_max[i][0]) / ( (config.column_min_max[i][1]) - (config.column_min_max[i][0]))

    return data


def pre_process():

    store_data = pd.read_csv(config.fileName)

    store_data = store_data.drop(store_data[(store_data.Open == 0) & (store_data.Sales == 0)].index)

    # store_data = store_data.drop(store_data[(store_data.Open != 0) & (store_data.Sales == 0)].index)

    # ---for segmenting original data ---------------------------------
    original_data = store_data.copy()

    # test_len = len(store_data[(store_data.Month == 7) & (store_data.Year == 2015)].index)
    #
    # train_size = int(len(store_data) - test_len)

    # train_data = store_data[:train_size]
    # original_data = original_data[train_size:]

    # -------------- processing train data---------------------------------------

    scaled_train_data = scale(store_data)
    train_X, train_y = segmentation(scaled_train_data)

    # ----segmenting original test data-----------------------------------------------

    nonescaled_X, nonescaled_y = segmentation(original_data)

    return train_X, train_y, nonescaled_y


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def RMSPE(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.sqrt(np.mean(np.square(((y_true - y_pred) / y_pred)), axis=0))

def plot(nonescaled_y,prediction_vals,RMSE_mean, name1, name2):

    fig1 = plt.figure()
    fig1 = plt.figure(dpi=100, figsize=(20, 7))
    days = range(len(RMSE_mean))
    plt.plot(days, RMSE_mean, label='RMSE mean')
    plt.legend(loc='upper left', frameon=False)
    plt.xlabel("day")
    plt.ylabel("RMSE")
    # plt.ylim((min(test_y), max(test_y)))
    plt.grid(ls='--')
    plt.savefig(name1, format='png', bbox_inches='tight', transparent=False)
    plt.close()

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
    plt.savefig(name2, format='png', bbox_inches='tight', transparent=False)
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

    test_pred[0] = (test_pred[0] * (config.column_min_max[0][1] - config.column_min_max[0][0])) + config.column_min_max[0][0]

    return test_pred

def SGD():

    predictions = []
    prediction_RMSE = []
    RMSE_mean = []

    train_X, train_y, nonescaled_y = pre_process()

    clf = linear_model.SGDRegressor()

    init_X = np.zeros(shape=(config.num_steps*config.features,)*1)
    init_X = init_X.reshape(1,-1)
    init_y = np.zeros(shape=(1,)*2)

    clf.partial_fit(init_X, init_y)

    for i in range(len(train_X)):
        tr_X, tr_y, none_y = train_X[i:i + 1], train_y[i:i + 1], nonescaled_y[i:i + 1]
        predsgdr = clf.predict(tr_X)
        clf.partial_fit(tr_X, tr_y)
        pred_val = rescle(predsgdr)
        predictions.append(pred_val[0])
        meanSquaredError = mean_squared_error(none_y[0], pred_val)
        rootMeanSquaredError = sqrt(meanSquaredError)
        print("RMSE:", rootMeanSquaredError)
        prediction_RMSE.append(rootMeanSquaredError)
        RMSE_mean.append(np.mean(prediction_RMSE))

    nonescaled_y= nonescaled_y.flatten()
    nonescaled_y= nonescaled_y.tolist()
    plot(nonescaled_y, predictions, RMSE_mean, "Sales RMSE mean SGDR mv online ML 2 store 165 .png", "sales price Prediction VS Truth SGDR mv online ML 2 store 165.png")

def PAR():

    predictions2 = []
    prediction_RMSE = []
    RMSE_mean = []

    train_X, train_y, nonescaled_y = pre_process()
    clf = linear_model.PassiveAggressiveRegressor()

    init_X = np.zeros(shape=(config.num_steps*config.features,)*1)
    init_X = init_X.reshape(1,-1)
    init_y = np.zeros(shape=(1,)*2)

    clf.partial_fit(init_X, init_y)

    for i in range(len(train_X)):
        tr_X, tr_y,none_y = train_X[i:i + 1], train_y[i:i + 1], nonescaled_y[i:i + 1]
        predsgdr = clf.predict(tr_X)
        clf.partial_fit(tr_X, tr_y)
        pred_val = rescle(predsgdr)
        predictions2.append(pred_val[0])
        meanSquaredError = mean_squared_error(none_y[0], pred_val)
        rootMeanSquaredError = sqrt(meanSquaredError)
        print("RMSE:", rootMeanSquaredError)
        prediction_RMSE.append(rootMeanSquaredError)
        RMSE_mean.append(np.mean(prediction_RMSE))

    nonescaled_y= nonescaled_y.flatten()
    plot(nonescaled_y, predictions2, RMSE_mean, "Sales RMSE mean PAR mv online ML 2 store 165 .png", "sales price Prediction VS Truth PAR mv online ML 2 store 165.png")




if __name__ == '__main__':
    SGD()
    PAR()
