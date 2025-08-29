from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from linear_regression import LinearRegression


def main():
    # load and download
    allData = fetch_california_housing()
    X = allData.data  # x contains the data
    y = allData.target  # y contains target values

    lr = LinearRegression()
    RMSE_test = myImplementation(X, y, lr)
    print("The Root Mean Squared Error of test data is", RMSE_test)

    listOfRMSE = []
    for i in range(20):
        listOfRMSE.append(myImplementation(X, y, lr))

    meanRMSE = lr.getMean(listOfRMSE)
    stdRMSE = lr.getStd(listOfRMSE)
    print("The mean of the Root Mean Squared Error is: ", meanRMSE)
    print("The standard deviation of the Root Mean Squared Error is: ", stdRMSE)


def myImplementation(X, y, lr):  # my implementation

    # 70% for training, 30% for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    lr.fit(X_test, y_test)
    yEst_test, MSE_test = lr.evaluate(X_test, y_test)
    RMSE_test = MSE_test ** 0.5

    return RMSE_test


if __name__ == "__main__":
    main()
