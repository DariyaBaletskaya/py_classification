import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
Y = np.array([4.3,  5.6,  6.9,  8,  9.9,  11.2,  12.9,  14.8,  15.5])

# Mean X and Y
mean_x = np.mean(X)
mean_y = np.mean(Y)

# Total number of values
N = len(X)

# Using the formula to calculate 'm' and 'c'
numer = 0
denom = 0
rmse = 0
rmse_plot_data = np.array([])
for i in range(N):
    numer += (X[i] - mean_x) * (Y[i] - mean_y)
    denom += (X[i] - mean_x) ** 2
    m = numer / denom
    c = mean_y - (m * mean_x)
    print("Coefficients")
    print(m, c)
    # Plotting Values and Regression Line

    max_x = np.max(X) + 100
    min_x = np.min(X) - 100

    # Calculating line values x and y
    x = np.linspace(min_x, max_x, 1000)
    y = c + m * x
    # Ploting Line
    plt.xlim(0, 10)
    plt.ylim(0, 17.5)
    # Ploting Scatter Points
    plt.scatter(X, Y, c='#ef5423', label='Scatter Plot')
    plt.plot(x, y, color='#' + '%06X' %
             random.randint(0, 0xFFFFFF), label='Regression Line')
    # Calculating Root Mean Squares Error
    y_pred = c + m * X[i]
    rmse += (Y[i] - y_pred) ** 2
    rmse = np.sqrt(rmse/N)
    rmse_plot_data = np.append(rmse_plot_data, rmse)
    plt.show()
    print("RMSE")
    print(rmse)

# Plot error values
plt.plot(X, rmse_plot_data)


plt.show()


# https://www.edureka.co/blog/least-square-regression/ <- Pay attention
# https://towardsdatascience.com/linear-regression-using-least-squares-a4c3456e8570
