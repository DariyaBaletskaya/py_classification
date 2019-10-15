import numpy as np
import matplotlib.pyplot as plt


class LinearRegressionUsingGD:
    XX = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0]])
    YY = np.array([[4.3], [5.6], [6.9], [8.0], [9.9], [11.2], [12.9], [14.8], [15.5]])

    def __init__(self, eta=0.05, n_iterations=10):
        self.eta = eta
        self.n_iterations = n_iterations

    def fit(self, x, y):
        self.cost_ = []
        self.w_ = np.zeros((x.shape[1], 1))
        m = x.shape[0]

        plt.ion()
        plt.show(block=False)

        for _ in range(self.n_iterations):
            y_pred = np.dot(x, self.w_)
            residuals = y_pred - y
            gradient_vector = np.dot(x.T, residuals)
            self.w_ -= (self.eta / m) * gradient_vector
            cost = np.sum((residuals ** 2)) / (2 * m)
            self.cost_.append(cost)

            plt.scatter(self.XX, self.YY)
            plt.text(2, 12, "Iteration: %d" % (_))
            plt.text(2, 13, "Cost: %s" % (cost))
            plt.pause(0.0000001)
            plt.clf()
            plt.plot(x, y_pred, '-g')
            plt.show(block=False)

        plt.close()
        return self

    def predict(self, x):
        return np.dot(x, self.w_)


def scatter_plot(x, y, size=10, x_label='x', y_label='y', color='b'):
    plt.scatter(x, y, s=size, color=color)
    set_labels(x_label, y_label)


def plot(x, y, x_label='x', y_label='y', color='r'):
    plt.plot(x, y, color=color)
    set_labels(x_label, y_label)


def ploty(y, x_label='x', y_label='y'):
    plt.plot(y)
    set_labels(x_label, y_label)


def set_labels(x_label, y_label):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
    plt.pause(3)
    plt.close()


class PerformanceMetrics:
    def __init__(self, y_actual, y_predicted):
        self.y_actual = y_actual
        self.y_predicted = y_predicted

    def compute_rmse(self):
        return np.sqrt(self.sum_of_square_of_residuals())

    def compute_r2_score(self):
        # sum of square of residuals
        ssr = self.sum_of_square_of_residuals()
        # total sum of errors
        sst = np.sum((self.y_actual - np.mean(self.y_actual)) ** 2)
        return 1 - (ssr / sst)

    def sum_of_square_of_residuals(self):
        return np.sum((self.y_actual - self.y_predicted) ** 2)


if __name__ == "__main__":
    # initializing the model
    linear_regression_model = LinearRegressionUsingGD()

    # generate the data set
    x = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0]])
    y = np.array([[4.3], [5.6], [6.9], [8.0], [9.9], [11.2], [12.9], [14.8], [15.5]])

    # transform the feature vectors to include the bias term
    # adding 1 to all the instances of the training set.
    m = x.shape[0]
    x_train = np.c_[np.ones((m, 1)), x]

    # fit/train the model
    linear_regression_model.fit(x_train, y)

    # predict values
    predicted_values = linear_regression_model.predict(x_train)

    # model parameters
    print(linear_regression_model.w_)
    intercept, coeffs = linear_regression_model.w_

    # cost_function
    cost_function = linear_regression_model.cost_

    # plotting
    scatter_plot(x, y)
    plot(x, predicted_values)
    ploty(cost_function, 'no of iterations', 'cost function')

    # computing metrics
    metrics = PerformanceMetrics(y, predicted_values)
    rmse = metrics.compute_rmse()
    r2_score = metrics.compute_r2_score()

    print('The coefficient is {}'.format(coeffs))
    print('The intercept is {}'.format(intercept))
    print('Root mean squared error of the model is {}.'.format(rmse))
    print('R-squared score is {}.'.format(r2_score))
