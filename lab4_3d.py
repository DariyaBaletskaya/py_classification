import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import axes3d


def f_NN(m, x, c):
    return m * x + c

if __name__ == "__main__":

    # Init data
    X = np.arange(1, 10, 1)
    Y_real = np.array([4.3, 5.6, 6.9, 8.0, 9.9, 11.2, 12.9, 14.8, 15.5])

    s1 = len(X) * sum(X * Y_real)
    s2 = sum(X) * sum(Y_real)
    s3 = len(X) * sum(X ** 2)
    s4 = sum(X) ** 2
    m_min_error = (s1 - s2) / (s3 - s4)
    c_min_error = (sum(Y_real) - m_min_error * sum(X)) / len(X)
    s5 = Y_real - (m_min_error * X + c_min_error)
    min_error = sum(s5 ** 2)

    n_epochs = 300
    m1 = 0.98
    m2 = 1.23
    dm = (m2 - m1) / n_epochs
    m_sf = np.arange(m1, m2, dm)

    c1 = 0.64
    c2 = 2.45
    dc = (c2 - c1) / n_epochs
    c_sf = np.arange(c1, c2, dc)
    E_sf = np.zeros([n_epochs, n_epochs])
    for i in range(n_epochs):
        mi = m_sf[i]
        for j in range(n_epochs):
            cj = c_sf[j]
            Eij = sum((Y_real - (mi * X + cj)) ** 2)
            E_sf[i, j] = Eij

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(m_sf, c_sf, E_sf, cmap='Greens_r')
    ax.set_xlabel('m')
    ax.set_ylabel('c')
    ax.set_zlabel('E')
    plt.show()
