import numpy as np
import matplotlib.pylab as plt

# matrix representation of letters
D_MATRIX = np.array([[-1, -1, -1, 1, -1, -1, -1],
                     [- 1, -1, 1, -1, 1, -1, -1],
                     [-1, 1, -1, -1, -1, 1, -1],
                     [-1, 1, -1, -1, -1, 1, -1],
                     [-1, 1, -1, -1, -1, 1, -1],
                     [-1, 1, -1, -1, -1, 1, -1],
                     [1, 1, 1,  1, 1, 1,     1],
                     [1, -1, -1, -1, -1, -1, 1],
                     [1, -1, -1, -1, -1, -1, 1], ])

B_MATRIX = np.array([[1, 1, 1, 1, 1, 1, 1],
                     [1, -1, -1, -1, -1, -1, -1],
                     [1, -1, -1, -1, -1, -1, -1],
                     [1, -1, -1, -1, -1, -1, -1],
                     [1, 1, 1, 1, 1, 1, 1],
                     [1, -1, -1, -1, -1, -1, 1],
                     [1, -1, -1, -1, -1, -1, 1],
                     [1, -1, -1, -1, -1, -1, 1],
                     [1, 1, 1, 1, 1, 1, 1], ])
I_MATRIX = np.array([[1, -1, -1, -1, -1, -1, 1],
                     [1, -1, -1, -1, -1, -1, 1],
                     [1, -1, -1, -1, -1, -1, 1],
                     [1, -1, -1, -1, -1, 1, 1],
                     [1, -1, -1, -1, 1, -1, 1],
                     [1, -1, -1, 1, -1, -1, 1],
                     [1, -1, 1,  -1, -1, -1, 1],
                     [1, 1, -1, -1, -1, -1, 1],
                     [1, -1, -1, -1, -1, -1, 1], ])

U_MATRIX = np.array([[1, -1, -1, -1, -1, -1, 1],
                     [-1, 1, -1, -1, -1, 1, -1],
                     [-1, -1, 1, -1, 1, -1, -1],
                     [-1, -1, -1, 1, -1, -1, -1],
                     [-1, -1, -1, 1, -1, -1, -1],
                     [-1, -1, 1, -1, -1, -1, -1],
                     [-1, -1, 1, -1, -1, -1, -1],
                     [-1, 1, -1, -1, -1, -1, -1],
                     [1, -1, -1, -1, -1, -1, -1], ])


def get_weight_matrix(matrix):
    matrix_flatten = np.reshape(matrix, (63, 1))
    matrix_flatten_transpose = matrix_flatten.transpose()
    weight_matrix = matrix_flatten.dot(matrix_flatten_transpose)

    return weight_matrix


def get_flatten_matrix(matrix):
    return np.reshape(matrix, (63, 1)).transpose()


def get_feature_matrix(first, second):
    result = np.array([[first[i][j] + second[i][j]
                        for j in range(len(first[0]))] for i in range(len(first))])
    np.fill_diagonal(result, 0)
    return result


def plot_letter(figure, matrix, plot_position):
    ax = figure.add_subplot(
        plot_position[0], plot_position[1], plot_position[2])
    ax.set_aspect('equal')
    plt.imshow(matrix, interpolation='nearest')


def train(matrix_flatten, feature_matrix):
    result = matrix_flatten.dot(feature_matrix)
    result = np.sign(result)
    result = np.reshape(result, (9, 7))
    return result


def Hopfield_NN():
    fig = plt.figure()
    plt.title("Source data")
    plot_letter(fig, D_MATRIX, [2, 2, 1])
    plot_letter(fig, B_MATRIX, [2, 2, 2])
    plot_letter(fig, I_MATRIX, [2, 2, 3])
    plot_letter(fig, U_MATRIX, [2, 2, 4])

    first_subset = get_feature_matrix(
        get_weight_matrix(D_MATRIX), get_weight_matrix(B_MATRIX))
    second_subset = get_feature_matrix(
        get_weight_matrix(I_MATRIX), get_weight_matrix(U_MATRIX))
    feature_matrix = get_feature_matrix(first_subset, second_subset)

    d_matrix_neural = train(get_flatten_matrix(D_MATRIX), feature_matrix)
    b_matrix_neural = train(get_flatten_matrix(B_MATRIX), feature_matrix)
    i_matrix_neural = train(get_flatten_matrix(I_MATRIX), feature_matrix)
    u_matrix_neural = train(get_flatten_matrix(U_MATRIX), feature_matrix)

    fig_neural = plt.figure()
    plt.title("Hopfield NN")
    plot_letter(fig_neural, d_matrix_neural, [2, 2, 1])
    plot_letter(fig_neural, b_matrix_neural, [2, 2, 2])
    plot_letter(fig_neural, i_matrix_neural, [2, 2, 3])
    plot_letter(fig_neural, u_matrix_neural, [2, 2, 4])
    plt.show()


if __name__ == "__main__":
    Hopfield_NN()
