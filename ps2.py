# Nate Ginck
# HW 2: Intro to ML

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from computeCost import computeCost
from gradientDescent import gradientDescent
from normalEqn import normalEqn

# 4a: read data1
autodf = pd.read_csv('input/hw2_data1.csv', header=None)

# 4b: plot scatterplot
plt.scatter(autodf[0], autodf[1], c="red", marker="x")
plt.xlabel("Horse power of a car in 100s")
plt.ylabel("Price in $1,000s")
plt.savefig('output/ps2-4-b.png')
plt.close()

# 4c: split auto into X and y
# convert to np array and add 1s column
auto = autodf.values
auto = np.insert(auto, 0, 1, axis=1)

# define X and y
X = auto[:, :2]
y = auto[:, 2]

# print size of X and y
print("Size of X: ", X.shape)
print("Size of y: ", y.shape)

# 4d: randomly divide data
np.random.shuffle(auto)

# take top 90% for training
split = int(auto.shape[0] * 0.9)

# split data using this value
# training data
X_train = auto[:split, :auto.shape[1] - 1]
y_train = auto[:split, auto.shape[1] - 1]

# testing data
X_test = auto[split:, :auto.shape[1] - 1]
y_test = auto[split:, auto.shape[1] - 1]

# 4e: compute gradient descent
theta, cost = gradientDescent(X_train, y_train, 0.3, 500)

# create and save plot
plt.plot(range(len(cost)), cost)
plt.title("Iterations vs Cost")
plt.savefig('output/ps2-4-e.png')
plt.close()

# print theta
print("Theta: ", theta)

# 4f: plot line of learned model
# find min and max
min = np.min(autodf[0])
max = np.max(autodf[0])

# use y = b0 + b1x to fit line of data
x_axis = np.linspace(min, max, 1000)
y_axis = theta[0] + x_axis * theta[1]

# plot and save the line
plt.scatter(autodf[0], autodf[1], c="lightblue", marker="x")
plt.plot(x_axis, y_axis, color="lightblue", label="Line")
plt.xlabel("Horse Power")
plt.ylabel("Price")
plt.savefig("output/ps2-4-f.png")
plt.close()

# 4g: find error on y_test

# regress model onto test set
h = theta * X_test
h = h.sum(axis=1)

#calculate error
E = h - y_test

# calculate squared error
SE = np.square(E)

# calculate sum squared error
SSE = np.sum(SE)

# calculate MSE and print
MSE = SSE/(2*X_test.shape[0])
print("MSE: ", MSE)

# 4h: use normalEqn to gather parameters
theta_norm = normalEqn(X_train, y_train)

# regress model onto test set
h2 = theta_norm * X_test
h2 = h2.sum(axis=1)

# calculate error
E2 = h2 - y_test

# calculate squared error
SE2 = np.square(E2)

# calculate sum squared error
SSE2 = np.sum(SE2)

# calculate MSE and print
MSE2 = SSE2/(2*X_test.shape[0])
print("MSE of normalized equation: ", MSE2)

# 4i: study effects of learning rate
# compute gradient descent
theta1, cost1 = gradientDescent(X_train, y_train, 0.001, 300)
theta2, cost2 = gradientDescent(X_train, y_train, 0.003, 300)
theta3, cost3 = gradientDescent(X_train, y_train, 0.03, 300)
theta4, cost4 = gradientDescent(X_train, y_train, 3, 300)

# create and save plots
plt.plot(range(len(cost1)), cost1)
plt.title("Iterations vs Cost")
plt.savefig('output/ps2-4-i-1.png')
plt.close()

plt.plot(range(len(cost2)), cost2)
plt.title("Iterations vs Cost")
plt.savefig('output/ps2-4-i-2.png')
plt.close()

plt.plot(range(len(cost3)), cost3)
plt.title("Iterations vs Cost")
plt.savefig('output/ps2-4-i-3.png')
plt.close()

plt.plot(range(len(cost4)), cost4)
plt.title("Iterations vs Cost")
plt.savefig('output/ps2-4-i-4.png')
plt.close()

# 5a. load data
co2df = pd.read_csv('input/hw2_data3.csv', header=None)
co2 = co2df.values
co2 = co2.astype(np.float64)

# find mean and sd of each predictor
mean0 = np.average(co2[0])
sd0 = np.std(co2[0])
mean1 = np.average(co2[1])
sd1 = np.std(co2[1])

# calculate normalization
co2[:, 0] = (co2[:, 0] - mean0)/sd0
co2[:, 1] = (co2[:, 1] - mean1)/sd1

# insert col of 1s
co2 = np.insert(co2, 0, 1, axis=1)

# split co2 into X and y
co2X = co2[:, :3]
co2y = co2[:, 3]

# print mean and sd of each vector, along with size
print("Mean and Standard Deviation of engine size: ", mean0, " and ", sd0)
print("Mean and Standard Deviation of car weight: ", mean1," and ", sd1)
print("The size of matrix X: ", co2X.shape)
print("The size of matrix y: ", co2y.shape)

# 5b: use gradient descent to calculate parameters
co2theta, co2cost = gradientDescent(co2X, co2y, 0.01, 750)

# plot
plt.plot(range(len(co2cost)), co2cost)
plt.title("Iterations vs Cost")
plt.savefig('output/ps2-5-b.png')
plt.close()

# print parameter theta
print("CO2 Theta: ", co2theta)

# 5c: predict CO2 emissions
# adjust parameter values
size_adj = (2300 - mean0)/sd0
weight_adj = (1300 - mean1)/sd1

y_pred = co2theta[0] + co2theta[1]*size_adj + co2theta[2]*weight_adj
print("Predicted value: ", y_pred)