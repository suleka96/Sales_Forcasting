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
    features = 1
    test_ratio = 0.2
    graph = tf.Graph()
    column1_min = 0
    column1_max = 10000
    column1 = 'Sales'
    fileName = 'store165_2.csv'


config = MLConfig()


def segmentation(data):

    seq = [price for tup in data[[config.column1]].values for price in tup]

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

    y = np.asarray(y)

    return X, y

def scale(data):

    data[config.column1] = (data[config.column1] - config.column1_min) / (config.column1_max - config.column1_min)

    return data


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

    days = range(len(true_vals))
    plt.plot(days, true_vals, label='truth sales')
    plt.plot(days, pred_vals, label='pred sales')
    plt.yscale('log')
    plt.legend(loc='upper left', frameon=False)
    plt.xlabel("day")
    plt.ylabel(" salese")
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

    pred_vals = [(pred * (config.column1_max - config.column1_min)) + config.column1_min for pred in prediction]

    pred_vals = np.asarray(pred_vals)

    pred_vals = pred_vals.flatten()

    nonescaled_y = nonescaled_y.flatten()

    get_scores("---------Linear Regression----------",pred_vals, nonescaled_y)

    write_results(nonescaled_y, pred_vals, "LinearRegression_results uv.csv")

    plot(nonescaled_y,pred_vals,"Liner Regression uv Prediction Vs Truth.png")


def XGB():

    train_X, train_y, test_X, test_y, nonescaled_y = pre_process()

    # param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
    # num_round = 2
    #
    # xgb = XGBRegressor(learning_rate=0.02, max_depth= 5, subsample=0.5,colsample_bytree=0.5,n_estimators=50)

    train_y = train_y.flatten()

    # gsc = GridSearchCV(
    #     estimator= XGBRegressor(),
    #     param_grid={
    #         'learning_rate': (0.1, 0.01,0.75),
    #         'max_depth': range(2, 5),
    #         'subsample': (0.5, 1,0.1, 0.75),
    #         'colsample_bytree': (1, 0.1, 0.75),
    #         'n_estimators': (50, 100, 1000)
    #     },
    #     cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
    #
    # grid_result = gsc.fit(train_X, train_y)
    # best_params = grid_result.best_params_
    #
    # print("\nBest: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    # xgb = XGBRegressor(learning_rate=best_params["learning_rate"],
    #                       max_depth=best_params["max_depth"], subsample=best_params["subsample"],
    #                       colsample_bytree=best_params["colsample_bytree"], n_estimators=best_params["n_estimators"],
    #                       coef0=0.1, shrinking=True, tol=0.001, cache_size=200, verbose=False, max_iter=-1)

    xgb = XGBRegressor(  max_depth=10, subsample=0.7,
                          colsample_bytree=0.7,eta= 0.02,num_round = 3000)

    bst = xgb.fit(train_X,train_y)

    preds = bst.predict(test_X)

    pred_vals = [(pred * (config.column1_max - config.column1_min)) + config.column1_min for pred in preds]


    pred_vals = np.asarray(pred_vals)

    nonescaled_y = nonescaled_y.flatten()

    get_scores("---------XGBoost Regressor----------", pred_vals, nonescaled_y)

    write_results(nonescaled_y, pred_vals, "XGBoost Regressor uv_results.csv")

    plot(nonescaled_y, pred_vals, "XGBoost Regressor uv Prediction Vs Truth.png")


def Random_Forest_Regressor():

    train_X, train_y, test_X, test_y, nonescaled_y = pre_process()

    train_y = train_y.flatten()

    # sc = GridSearchCV(
    #     estimator=RandomForestRegressor(),
    #     param_grid={
    #         'max_depth': range(2, 5),
    #         'n_estimators': (50, 100, 1000)
    #     },
    #     cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
    #
    # grid_result = sc.fit(train_X, train_y)
    # best_params = grid_result.best_params_
    #
    # print("\nBest: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    # rfr = RandomForestRegressor(
    #     max_depth=best_params["max_depth"], n_estimators=best_params["n_estimators"], verbose=False)

    rfr = RandomForestRegressor(
            max_depth=35, n_estimators=200, min_samples_split=2,min_samples_leaf=1)

    rfr.fit(train_X, train_y)

    rfr_prediction = rfr.predict(test_X)

    pred_vals = [(pred * (config.column1_max - config.column1_min)) + config.column1_min for pred in rfr_prediction]

    pred_vals = np.asarray(pred_vals)

    nonescaled_y = nonescaled_y.flatten()

    get_scores("---------Random Forest Regressor----------", pred_vals, nonescaled_y)

    write_results(nonescaled_y, pred_vals, "RandomForestRegressor uv_results.csv")

    plot(nonescaled_y, pred_vals, "Random Forest Regressor uv Prediction Vs Truth.png")


if __name__ == '__main__':
    Linear_Regression()
    Random_Forest_Regressor()
    XGB()