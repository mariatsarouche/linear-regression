from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from linear_regression import LinearRegression
from sklearn.linear_model import LinearRegression as skLinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np


def main():
    # load and download
    allData = fetch_california_housing()
    X = allData.data  # x contains the data
    y = allData.target  # y contains target values
    myImplementation(X, y)
    skLearnImplementation(X, y)


def myImplementation(X, y):
    RMSE_test = rmse(X, y, 42)
    print("My implementation.\nThe Root Mean Squared Error of test data is", RMSE_test)

    listOfRMSE = []
    for i in range(20):
        listOfRMSE.append(rmse(X, y, i))

    meanRMSE = np.mean(listOfRMSE)
    stdRMSE = np.std(listOfRMSE)
    print("The mean of the Root Mean Squared Error is: ", meanRMSE)
    print("The standard deviation of the Root Mean Squared Error is: ", stdRMSE)


def rmse(X, y, seed):  # my implementation
    lr = LinearRegression()
    # 70% for training, 30% for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

    lr.fit(X_train, y_train)
    yEst_test, MSE_test = lr.evaluate(X_test, y_test)
    RMSE_test = MSE_test ** 0.5

    return RMSE_test


def skLearnImplementation(X, y):
    listOfRMSE = []
    for i in range(20):
        listOfRMSE.append(getRmse(X, y, i))

    meanRMSE = np.mean(listOfRMSE)
    stdRMSE = np.std(listOfRMSE)
    print("\nScikit-learn's implementation.\nThe mean of the Root Mean Squared Error is: ", meanRMSE)
    print("The standard deviation of the Root Mean Squared Error is: ", stdRMSE)


def getRmse(X, y, seed):
    sk = skLinearRegression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
    sk.fit(X_train, y_train)
    yEst = sk.predict(X_test)
    return mean_squared_error(y_test, yEst) ** 0.5


if __name__ == "__main__":
    main()
