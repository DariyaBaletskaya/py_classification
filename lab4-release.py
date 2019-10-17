import numpy as np
import matplotlib.pyplot as plt
import random


class LinRegressionLeastsquares:
    def setCoefficients(self, m_pred, c_pred):
        self.m_pred = m_pred
        self.c_pred = c_pred

    def calculate_error(self, y_predicted, y_actual):
        return y_predicted - y_actual

    def least_squares(self, X, Y):
       # Total number of values
        N = len(X)
        mean_x = np.mean(X)
        mean_y = np.mean(Y)

        # Using the formula to calculate 'm' and 'c'
        numer = 0
        denom = 0
        error = 0
        for i in range(N):
            numer += (X[i] - mean_x) * (Y[i] - mean_y)
            denom += (X[i] - mean_x) ** 2
            m = numer / denom
            c = mean_y - (m * mean_x)
            error += (Y[i] - (m * X[i] + c)) ** 2
        return m, c, error

    def train(self, X, Y, coef, err_min, iterations):
      # generate random coefficients
        m = np.random.uniform(-5, 5)
        c = np.random.uniform(-5, 5)

        errors = np.array([])
        for _ in range(iterations):
            iter_error = 0
            for i in range(len(X)):
                y_predicted = m * X[i] + c
                iter_error = self.calculate_error(Y[i], y_predicted)
                dm = iter_error * X[i] * coef
                dc = iter_error * coef
                m += dm
                c += dc
                y_predicted = m * X[i] + c
                iter_error += self.calculate_error(Y[i], y_predicted)

            plt.pause(0.01)
            plt.clf()
            Y_actual = [m * i + c for i in X]
            plt.plot(X, Y_actual, '-g')
            plt.text(2, 12, "Iteration: %d" % (_))
            plt.plot(X, Y, '.b')
            Y_predict = [self.m_pred * i + self.c_pred for i in X]
            plt.plot(X, Y_predict, '-r')

            errors = np.append(errors, abs(iter_error))

            if(errors[len(errors)-1]) <= err_min:
                break

        plt.clf()
        plt.plot(errors)
        plt.xlabel('Iterations')
        plt.ylabel('Errors')
        plt.pause(3)
        plt.show()


def main():
    linear_regression_model = LinRegressionLeastsquares()
    X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    Y = np.array([4.3,  5.6,  6.9,  8,  9.9,  11.2,  12.9,  14.8,  15.5])

    m_least_squares, c_least_squares, err_least_squares = linear_regression_model.least_squares(
        X, Y)
    linear_regression_model.setCoefficients(m_least_squares, c_least_squares)
    print("Coefficients")
    print(m_least_squares, c_least_squares, err_least_squares)
    Y_predict = [m_least_squares * i + c_least_squares for i in X]

    # plotting
    plt.plot(X, Y, '.b')
    plt.plot(X, Y_predict, '-r')
    plt.ion()
    plt.show()

    linear_regression_model.train(X, Y, 0.01, err_least_squares, 200)


if __name__ == '__main__':
    main()
