from statsmodels.tsa.arima_model import ARIMA
import pandas as pd
import matplotlib as mplt
mplt.use('agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pylab as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
import csv


class ARIMAConfig():
    lag_order = 1
    degree_differencing = 0
    order_moving_avg = 1
    test_ratio = 0.2
    fileName = 'store165_2.csv'
    min = 0
    max = 10000
    column = 'Sales'
    store = 285

config = ARIMAConfig()

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def scale(data):

    data[config.column] = (data[config.column] - config.min) / (config.max - config.min)

    return data

def plot(original_test_list,pred_vals):
    days = range(len(original_test_list))
    plt.plot(days, original_test_list, color='blue', label='truth sales')
    plt.plot(days, pred_vals, color='red', label='pred sales')
    plt.yscale('log')
    plt.legend(loc='upper left', frameon=False)
    plt.xlabel("day")
    plt.ylabel("closing price")
    plt.grid(ls='--')
    plt.savefig("Sales ARIMA Prediction Vs Truth .png", format='png', bbox_inches='tight', transparent=False)
    plt.close()

def preprocess():

    store_data = pd.read_csv(config.fileName)

    store_data = store_data.drop(store_data[(store_data.Open == 0) & (store_data.Sales == 0)].index)

    # store_data = store_data.drop(store_data[(store_data.Open != 0) & (store_data.Sales == 0)].index)

    store_data_scale = store_data
    store_data_orginal = store_data.copy()

    scaled_data = scale(store_data_scale)
    sales = scaled_data[config.column]

    # ---for segmenting original data ---------------------------------
    nonescaled_sales= store_data_orginal[config.column]

    # size = int(len(sales) * (1.0 - config.test_ratio))
    test_len = len(store_data[(store_data.Month == 7) & (store_data.Year == 2015)].index)

    size = int(len(store_data) - test_len)
    train = sales[:size]
    test = sales[size:]
    original_train = nonescaled_sales[:size]
    original_test = nonescaled_sales[size:]

    return train,test,original_train,original_test

def write_results(true_vals,pred_vals,name):
    with open(name, "w") as f:
        writer = csv.writer(f)
        writer.writerows(zip(true_vals, pred_vals))

def ARIMA_model():

    train, test, original_train, original_test = preprocess()

    predictions = list()
    history = [x for x in train]
    test_list= test.tolist()
    original_test_list = original_test.tolist()

    for t in range(len(test_list)):
        model = ARIMA(history, order=(config.lag_order, config.degree_differencing, config.order_moving_avg))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test_list[t]
        history.append(obs)
        print('predicted=%f, expected=%f' % (yhat, obs))

    pred_vals = [(pred[0] * (config.max - config.min)) + config.min for pred in predictions]

    meanSquaredError = mean_squared_error(original_test_list, pred_vals)
    rootMeanSquaredError = sqrt(meanSquaredError)
    print("RMSE:", rootMeanSquaredError)
    mae = mean_absolute_error(original_test_list, pred_vals)
    print("MAE:", mae)
    mape = mean_absolute_percentage_error(original_test_list, pred_vals)
    print("MAPE:", mape)

    write_results(original_test_list, pred_vals, "ARIMA_results.csv")

    plot(original_test_list,pred_vals)


if __name__ == '__main__':
    ARIMA_model()







