import numpy as np


def checkIfNpArray(array, name=None):
    if not isinstance(array, np.ndarray):
        raise ValueError(f"{name} isn't a numpy array.")


def rowsMatch(inp, out):
    if not inp.shape[0] == out.shape[0]:
        raise ValueError(f"Arrays don't have the same amount of rows.")


def check1D(y):
    if y.ndim != 1:
        raise ValueError(f"y must be 1D, it's {y.ndim}D.")


def verifyArrays(X, y):
    # check if both are numpy array
    checkIfNpArray(X, "X")
    checkIfNpArray(y, "y")
    rowsMatch(X, y)   # check if dimensions are compatible
    check1D(y)  # y must be 1d


def createNewX(X):
    # adds a row of 1 to x for bias
    return np.append(X, np.ones((X.shape[0], 1), dtype=int), axis=1)


def computeTH(X, y):
    # solve normal equations to find theta
    return np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))


def checkIfTrained(w, b):
    if w is None or b is None: # if w or b doesn't exist it's not trained
        raise ValueError(f"The model isn't trained.")


def computeMSE(N, yEst, y):
    err = yEst - y
    return np.dot(err.T, err) / N


class LinearRegression:

    def fit(self, X, y):
        verifyArrays(X, y)
        self.X = X
        self.y = y
        newX = createNewX(X)
        theta = computeTH(newX, y)
        self.w = theta[:-1]  # all except from last
        self.b = theta[-1]   # only last

    def predict(self, X):
        checkIfTrained(self.w, self.b)
        return np.dot(X, self.w) + self.b  # return predictions

    def evaluate(self, X, y):
        yEst = self.predict(X)  # yEst is estimations/predictions that are produced
        MSE = computeMSE(len(y), yEst, y)
        return yEst, MSE

    def getMean(self, listPar):
        self.listPar = listPar
        return np.mean(listPar)

    def getStd(self, listPar):
        self.listPar = listPar
        return np.std(listPar)

    def __init__(self):
        self.w = None
        self.b = None
        self.X = None
        self.y = None
        self.listPar = None
