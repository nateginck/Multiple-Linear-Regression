from computeCost import computeCost

import numpy as np
import pandas as pd


# X_train: learning data, y_train: response, alpha: learning rate, iters: iterations for gradient descent
def gradientDescent(X_train, y_train, alpha, iters):
    # create an empty array
    cost = np.zeros(iters)
    # initialize theta to a random value for n+1 x 1 vector
    theta = np.random.rand(X_train.shape[1])
    # create a temp value array for n+1 x 1 vector
    temp = np.empty(X_train.shape[1])

    for i in range(iters):
        # compute cost
        cost[i] = computeCost(X_train, y_train, theta)

        # for n + 1 features
        for j in range(X_train.shape[1]):
            # calculate temp values
            if j == 0:
                estimate = X_train * theta
                estimate = np.sum(estimate, axis=1)

                # compute error
                error = estimate - y_train

                # find cost derivative value
                dJ = np.sum(error) / (X_train.shape[0])

                temp[j] = theta[j] - alpha * dJ
            else:
                estimate = X_train * theta
                estimate = np.sum(estimate, axis=1)

                # compute error
                error = estimate - y_train
                error = error.reshape(-1, 1) * X_train

                # find cost derivative value
                dJ = np.sum(error) / (X_train.shape[0])

                temp[j] = theta[j] - alpha * dJ

        # change the estimates to new values
        theta = temp

    return theta, cost


if '__main__' == __name__:
    # enter toy dataset
    toy = np.array([[1, 1, 1, 8], [1, 2, 2, 6], [1, 3, 3, 4], [1, 4, 4, 2]])

    # print estimates for theta, and cost over iterations
    theta, cost = gradientDescent(toy[:, 0:3], toy[:, 3], 0.001, 15)
    print("Theta: ", theta)
    print("Cost: ", cost)
