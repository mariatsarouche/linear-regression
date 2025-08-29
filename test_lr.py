from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


def main():
    # load and download
    allData = fetch_california_housing()
    X = allData.data  # x contains the data
    y = allData.target  # y contains target values

    # 70% for training, 30% for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


if __name__ == "__main__":
    main()