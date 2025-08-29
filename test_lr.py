from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from linear_regression import LinearRegression
from sklearn.linear_model import LinearRegression as skLinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


def main():
    # load and download
    allData = fetch_california_housing()
    X = allData.data  # x contains the data
    y = allData.target  # y contains target values
    myImplementation(X, y)
    skLearnImplementation(X, y)


def myImplementation(X, y):
    lr = LinearRegression()
    RMSE_test = rmse(X, y, lr, 42)
    print("My implementation.\nThe Root Mean Squared Error of test data is", RMSE_test)

    listOfRMSE = []
    for i in range(20):
        listOfRMSE.append(rmse(X, y, lr, i))

    meanRMSE = np.mean(listOfRMSE)
    stdRMSE = np.std(listOfRMSE)
    print("The mean of the Root Mean Squared Error is: ", meanRMSE)
    print("The standard deviation of the Root Mean Squared Error is: ", stdRMSE)


def rmse(X, y, lr, seed):  # my implementation
    # 70% for training, 30% for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

    lr.fit(X_train, y_train)
    yEst_train, MSE_train = lr.evaluate(X_train, y_train)
    RMSE_train = MSE_train ** 0.5

    return RMSE_train


def skLearnImplementation(X, y):
    sk = skLinearRegression()
    listOfRMSE = []
    for i in range(20):
        listOfRMSE.append(getRmse(X, y, i, sk))

    meanRMSE = np.mean(listOfRMSE)
    stdRMSE = np.std(listOfRMSE)
    print("\nScikit-learn's implementation.\nThe mean of the Root Mean Squared Error is: ", meanRMSE)
    print("The standard deviation of the Root Mean Squared Error is: ", stdRMSE)


def getRmse(X, y, seed, sk):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
    sk.fit(X_train, y_train)
    yEst = sk.predict(X_test)
    return mean_squared_error(y_test, yEst) ** 0.5


if __name__ == "__main__":
    main()
