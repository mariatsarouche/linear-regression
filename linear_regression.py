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

def verifyArray(X, y):
    checkIfNpArray(X, "X")
    checkIfNpArray(y, "y")
    rowsMatch(X, y)
    check1D(y)


class LinearRegression:

    def fit(self, X, y):
        verifyArray(X, y)

    def predict(self, X):


    def evaluate(self, X, y):


    def __init__(self):
        self.w = None
        self.b = None