import tensorflow as tf
import matplotlib as mplt
mplt.use('agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from math import sqrt
from sklearn.linear_model import LinearRegression
import csv
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV


class MLConfig():
    input_size = 1
    num_steps = 8
    lstm_size = 32
    num_layers = 1
    keep_prob = 0.8
    batch_size = 16
    init_learning_rate = 0.01
    learning_rate_decay = 0.99
    init_epoch = 10  # 5
    max_epoch = 50  # 100 or 50
    test_ratio = 0.2
    fileName = 'store165_3.csv'
    graph = tf.Graph()
    features = 5
    column_min_max = [[0, 10000], [1, 7]]
    columns = ['Sales', 'DayOfWeek', 'SchoolHoliday', 'Promo', 'Lagged_Open']


config = MLConfig()


def segmentation(data):

    seq = [price for tup in data[config.columns].values for price in tup]

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
    y = [y[i][0] for i in range(len(y))]

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

    store_data = pd.read_csv(config.fileName)

    store_data = store_data.drop(store_data[(store_data.Open == 0) & (store_data.Sales == 0)].index)

    # store_data = store_data.drop(store_data[(store_data.Open != 0) & (store_data.Sales == 0)].index)

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

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def plot(true_vals,pred_vals,name):

    fig = plt.figure()
    fig = plt.figure(dpi=100, figsize=(20, 7))
    days = range(len(true_vals))
    plt.plot(days, true_vals, label='truth sales')
    plt.plot(days, pred_vals, label='pred sales')
    plt.yscale('log')
    plt.legend(loc='upper left', frameon=False)
    plt.xlabel("day")
    plt.ylabel("sales")
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


def RMSPE(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.sqrt(np.mean(np.square(((y_true - y_pred) / y_pred)), axis=0))

def write_results(true_vals,pred_vals,name):

    true_vals=true_vals.tolist()
    pred_vals=pred_vals.tolist()

    with open(name, "w") as f:
        writer = csv.writer(f)
        writer.writerows(zip(true_vals, pred_vals))

def Linear_Regression():

    train_X, train_y, test_X, test_y, nonescaled_y = pre_process()

    reg = LinearRegression().fit(train_X, train_y)

    prediction = reg.predict(test_X)

    # pred_vals, nonescaled_y = prediction, test_y

    pred_vals = rescle(prediction)

    pred_vals = np.asarray(pred_vals)

    pred_vals = pred_vals.flatten()

    nonescaled_y = nonescaled_y.flatten()

    get_scores("---------Linear Regression----------",pred_vals, nonescaled_y)

    write_results(nonescaled_y, pred_vals, "LinearRegression_mv_results.csv")

    plot(nonescaled_y,pred_vals,"Liner Regression mv Prediction Vs Truth.png")


def XGB():

    train_X, train_y, test_X, test_y, nonescaled_y = pre_process()

    # param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
    # num_round = 2
    #
    # xgb = XGBRegressor(learning_rate=0.02, max_depth= 5, subsample=0.5,colsample_bytree=0.5,n_estimators=50)

    gsc = GridSearchCV(
        estimator= XGBRegressor(),
        param_grid={
            'learning_rate': (0.1, 0.01,0.75),
            'max_depth': range(2, 5),
            'subsample': (0.5, 1,0.1, 0.75),
            'colsample_bytree': (1, 0.1, 0.75),
            'n_estimators': (50, 100, 1000)
        },
        cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)

    grid_result = gsc.fit(train_X, train_y)
    best_params = grid_result.best_params_

    print("\nBest: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    xgb = XGBRegressor(learning_rate=best_params["learning_rate"],
                          max_depth=best_params["max_depth"], subsample=best_params["subsample"],
                          colsample_bytree=best_params["colsample_bytree"], n_estimators=best_params["n_estimators"],
                          coef0=0.1, shrinking=True, tol=0.001, cache_size=200, verbose=False, max_iter=-1)

    bst = xgb.fit(train_X,train_y)

    preds = bst.predict(test_X)

    pred_vals = rescle(preds)

    pred_vals = np.asarray(pred_vals)

    nonescaled_y = nonescaled_y.flatten()

    get_scores("---------XGBoost Regressor----------", pred_vals, nonescaled_y)

    write_results(nonescaled_y, pred_vals, "XGBoost Regressor mv_results.csv")

    plot(nonescaled_y, pred_vals, "XGBoost Regressor mv Prediction Vs Truth.png")


def Random_Forest_Regressor():

    train_X, train_y, test_X, test_y, nonescaled_y = pre_process()

    gsc = GridSearchCV(
        estimator=RandomForestRegressor(),
        param_grid={
            'max_depth': range(2, 5),
            'n_estimators': (50, 100, 1000)
        },
        cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)

    grid_result = gsc.fit(train_X, train_y)
    best_params = grid_result.best_params_

    print("\nBest: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    rfr = RandomForestRegressor(
                       max_depth=best_params["max_depth"], n_estimators=best_params["n_estimators"],verbose=False)

    rfr.fit(train_X, train_y)

    rfr_prediction = rfr.predict(test_X)

    pred_vals = rescle(rfr_prediction)

    pred_vals = np.asarray(pred_vals)

    nonescaled_y=nonescaled_y.flatten()

    get_scores("---------Random Forest Regressor----------",pred_vals,nonescaled_y)

    write_results(nonescaled_y, pred_vals, "RandomForestRegressor mv_results.csv")

    plot(nonescaled_y,pred_vals,"Random Forest Regressor mv Prediction Vs Truth.png")


if __name__ == '__main__':
    Linear_Regression()
    Random_Forest_Regressor()
    XGB()