# https://machinelearningmastery.com/implement-simple-linear-regression-scratch-python/

from math import sqrt


def mean(x):
    return sum(x) / len(x)


def variance(x):
    mean_ = mean(x)
    return sum([(i - mean_)**2 for i in x])


def covariance(x, y):
    x_mean = mean(x)
    y_mean = mean(y)
    covar = 0.0
    for i in range(len(x)):
        covar += (x[i] - x_mean) * (y[i] - y_mean)
    return covar


def coefs(X):
    x = [row[0] for row in X]
    y = [row[1] for row in X]
    x_mean, y_mean = mean(x), mean(y)
    b1 = covariance(x, y) / variance(x)
    b0 = y_mean - b1 * x_mean
    return [b0, b1]


def rmse(true, predicted):
    loss = 0.0
    for i in range(len(true)):
        loss += ((predicted[i] - true[i])**2)
    return sqrt(loss/len(true))


def linear_regression(train, test):
    predictions = []
    b0, b1 = coefs(train)
    for row in test:
        predictions.append(b0 + b1 * row[0])
    return predictions


dataset = [[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]]
