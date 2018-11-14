import pandas as pd
import matplotlib.pylab as plt



def drawSubplots():


    linearreg_true, linearreg_pred  = read_csv("LinearRegression_results.csv")

    rfr_true, rfr_pred = read_csv("RandomForestRegressor_results.csv")

    arima_true, arima_pred = read_csv("ARIMA_results.csv")

    rnn_true, rnn_pred = read_csv("RNN_results.csv")

    size=10

    fig = plt.figure()

    fig = plt.figure(dpi=100, figsize=(20, 7))

    plt.subplot(221)
    days = range(len(linearreg_true))
    plt.plot(days, linearreg_true,color='r', label='truth sales' )
    plt.plot(days, linearreg_pred,color='b', label='pred sales')
    plt.yscale('log')
    plt.xlabel("days")
    plt.ylabel("sales")
    plt.legend(loc='upper left', frameon=False)
    plt.title('Linear Regression',fontsize=size)


    plt.subplot(222)
    days = range(len(rfr_true))
    plt.plot(days, rfr_true,color='r')
    plt.plot(days, rfr_pred,color='b')
    plt.yscale('log')
    plt.xlabel("days")
    plt.ylabel("sales")
    # plt.legend(loc='upper left', frameon=False)
    plt.title('Random Forest Regressor',fontsize=size)


    plt.subplot(223)
    days = range(len(arima_true))
    plt.plot(days, arima_true,color='r')
    plt.plot(days, arima_pred,color='b')
    plt.yscale('log')
    plt.xlabel("days")
    plt.ylabel("sales")
    # plt.legend(loc='upper left', frameon=False)
    plt.title('ARIMA',fontsize=size)


    plt.subplot(224)
    days = range(len(rnn_true))
    plt.plot(days, rnn_true,color='r' )
    plt.plot(days, rnn_pred,color='b' )
    plt.yscale('log')
    plt.xlabel("days")
    plt.ylabel("sales")
    # plt.legend(loc='upper left', frameon=False)
    plt.title('LSTM',fontsize=size)

    plt.savefig("store 285 subplots.png", format='png', bbox_inches='tight', transparent=False)

    plt.show()

    # plt.subplot
    #
    # f, axarr = plt.subplots(4, sharey=True)
    # f.suptitle('Sales Prediction store 285')
    #
    # days = range(len(linearreg_true))
    # axarr[0].plot(days, linearreg_true,color='r', label='truth sales' )
    # axarr[0].plot(days, linearreg_pred,color='b', label='pred sales')
    # axarr[0].plt.legend(loc='upper left', frameon=False)
    # axarr[0].plt.yscale('log')
    # axarr[0].plt.xlabel("days")
    # axarr[0].plt.ylabel("sales")
    # axarr[0].plt
    #
    #
    #
    # days = range(len(rfr_true))
    # axarr[1].plot(days, rfr_true, color='r', label='truth sales')
    # axarr[1].plot(days, rfr_pred, color='b', label='pred sales')
    # axarr[1].plt.legend(loc='upper left', frameon=False)
    # axarr[1].plt.yscale('log')
    # axarr[1].plt.xlabel("days")
    # axarr[1].plt.ylabel("sales")
    #
    # days = range(len(arima_true))
    # axarr[2].plot(days, arima_true, color='r', label='truth sales')
    # axarr[2].plot(days, arima_pred, color='b', label='pred sales')
    # axarr[2].plt.legend(loc='upper left', frameon=False)
    # axarr[2].plt.yscale('log')
    # axarr[2].plt.xlabel("days")
    # axarr[2].plt.ylabel("sales")
    #
    # days = range(len(rnn_true))
    # axarr[3].plot(days, rnn_true, color='r', label='truth sales')
    # axarr[3].plot(days, rnn_pred, color='b', label='pred sales')
    # axarr[3].plt.legend(loc='upper left', frameon=False)
    # axarr[3].plt.yscale('log')
    # axarr[3].plt.xlabel("days")
    # axarr[3].plt.ylabel("sales")
    #
    # f.plt.savefig("store 286 subplot", format='png', bbox_inches='tight', transparent=False)


def read_csv(filename):
    store_data = pd.read_csv(filename)
    x = store_data.iloc[:,0]
    y = store_data.iloc[:,1]

    return x,y

if __name__ == '__main__':
    drawSubplots()
