import numpy as np


def checkIfNpArray(array, name=None):
    if not isinstance(array, np.ndarray):
        raise ValueError(f"{name} isn't a numpy array.")


def rowsMatch(inp, out):
    if not inp.shape[0] == out.shape[0]:
        raise ValueError(f"Arrays dont have the same amount of rows.")


def check1D(y):
    if y.ndim != 1:
        raise ValueError(f"y must be 1D, it's {y.ndim}D")


def verifyArrays(X, y):
    checkIfNpArray(X, "X")
    checkIfNpArray(y, "y")
    rowsMatch(X, y)
    check1D(y)

def createNewX(X):
    return np.append(X, np.ones((X.shape[0], 1), dtype=int), axis=1)


def computeTH(X, y):
    return np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))


def checkIfTrained(w, b):
    if w is None or b is None:
        raise ValueError(f"The model isn't trained")
    
class LinearRegression:

    def fit(self, X, y):
        verifyArrays(X, y)
        self.X = X
        self.y = y
        newX = createNewX(X)
        theta = computeTH(newX, y)
        self.w = theta[:-1]
        self.b = theta[-1]

    def predict(self, X):
        checkIfTrained(self.w, self.b)
        return np.dot(X, self.w) + self.b


    def __init__(self):
        self.w = None
        self.b = None
        self.X = None
        self.y = None
