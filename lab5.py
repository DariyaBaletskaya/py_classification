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


# matrix representation of letter with noizes

D_MATRIX_NOISED = np.array([[-1, -1, -1, 1, 1, -1, -1],
                            [- 1, -1, 1, -1, 1, -1, -1],
                            [-1, 1, -1, -1, -1, 1, -1],
                            [-1, 1, -1, -1, -1, 1, -1],
                            [-1, 1, -1, -1, 1, 1, -1],
                            [-1, 1, -1, -1, -1, 1, -1],
                            [1, 1, 1,  1, 1, 1,     1],
                            [1, -1, -1, -1, -1, -1, 1],
                            [1, -1, -1, -1, -1, -1, 1], ])
B_MATRIX_NOISED = np.array([[1, 1, 1, 1, 1, 1, 1],
                            [1, -1, -1, -1, -1, -1, -1],
                            [1, -1, -1, -1, -1, -1, 1],
                            [1, -1, -1, -1, -1, -1, -1],
                            [1, 1, 1, 1, 1, 1, 1],
                            [1, -1, -1, -1, -1, -1, 1],
                            [1, -1, -1, -1, -1, -1, 1],
                            [1, -1, -1, 1, -1, -1, 1],
                            [1, 1, 1, 1, 1, 1, 1], ])
I_MATRIX_NOISED = np.array([[1, -1, -1, -1, -1, -1, 1],
                            [1, -1, -1, -1, -1, -1, 1],
                            [1, -1, -1, -1, -1, -1, 1],
                            [1, -1, -1, -1, -1, 1, 1],
                            [1, -1, -1, -1, 1, -1, 1],
                            [1, -1, -1, 1, -1, 1, 1],
                            [1, -1, 1,  1, -1, -1, 1],
                            [1, 1, -1, -1, -1, -1, 1],
                            [1, -1, -1, -1, -1, -1, 1], ])
U_MATRIX_NOISED = np.array([[1, -1, -1, -1, -1, -1, 1],
                            [-1, 1, -1, -1, -1, 1, -1],
                            [-1, -1, 1, -1, 1, -1, -1],
                            [-1, -1, -1, 1, -1, -1, -1],
                            [-1, -1, -1, 1, -1, 1, -1],
                            [-1, -1, 1, -1, -1, -1, -1],
                            [-1, -1, 1, -1, -1, -1, -1],
                            [-1, 1, -1, 1, -1, -1, -1],
                            [1, -1, -1, -1, -1, -1, -1], ])
#ascii binary letter representation
D_ASCII = np.array([[1,-1,1,-1,-1,1,-1,-1]])
B_ASCII = np.array([[-1,1,-1,-1,-1,-1,1,-1]])
I_ASCII = np.array([[-1,1,-1,-1,1,-1,-1,1]])
U_ASCII = np.array([[-1,1,-1,1,-1,1,-1,1]])


# calculate weight matrix for Hopfield NN
def get_weight_matrix_hopfield(matrix):
    matrix_flatten = np.reshape(matrix, (63, 1))
    matrix_flatten_transpose = matrix_flatten.transpose()
    weight_matrix = matrix_flatten.dot(matrix_flatten_transpose)

    return weight_matrix

# calculate weight matrix for BAM NN
# @param matrix - letter matrix
# @param sample - ascii code sample for autoassociation
def get_weight_matrix_bam(matrix, ascii_sample):
    matrix_flatten = np.reshape(matrix, (63, 1))
    weight_matrix = matrix_flatten.dot(ascii_sample)
    
    return weight_matrix
    
# flatten matrix into row
def get_flatten_matrix(matrix):
    return np.reshape(matrix, (63, 1)).transpose()

# feature matrix for Hopfield NN
# main diagonal should be filled with zeroes
def get_feature_matrix_hopfield(first, second):
    result = np.array([[first[i][j] + second[i][j]
                        for j in range(len(first[0]))] for i in range(len(first))])
    np.fill_diagonal(result, 0)
    return result

# feature matrix for BAM NN
def get_feature_matrix_bam(first, second):
    result = np.array([[first[i][j] + second[i][j]
                        for j in range(len(first[0]))] for i in range(len(first))])
    
    return result

#plotting letter from matrix representation
def plot_letter(figure, matrix, plot_position):
    ax = figure.add_subplot(
        plot_position[0], plot_position[1], plot_position[2])
    ax.set_aspect('equal')
    plt.imshow(matrix, interpolation='nearest')

#get Hopfield NN association of matrix letter representation
def associate_hopfield(matrix_flatten, feature_matrix):
    result = matrix_flatten.dot(feature_matrix)
    result = np.sign(result)
    result = np.reshape(result, (9, 7))
    return result

#get BAM NN association between ascii and matrix
def associate_ascii(ascii_matrix, feature_matrix):
    feature_matrix = feature_matrix.transpose()
    result = ascii_matrix.dot(feature_matrix)
    result = np.sign(result)
    result = np.reshape(result, (9, 7))
    print(result)
    return result

#get BAM NN association between matrix and ascii
def associate_matrix(matrix, feature_matrix):
    matrix_flatten = get_flatten_matrix(matrix)
    result = matrix_flatten.dot(feature_matrix)
    result = np.sign(result)
    return result


def Hopfield_NN():
    fig = plt.figure()
    plt.title("Source data")
    plot_letter(fig, D_MATRIX, [2, 2, 1])
    plot_letter(fig, B_MATRIX, [2, 2, 2])
    plot_letter(fig, I_MATRIX, [2, 2, 3])
    plot_letter(fig, U_MATRIX, [2, 2, 4])

    first_feature_subset = get_feature_matrix_hopfield(
        get_weight_matrix_hopfield(D_MATRIX), get_weight_matrix_hopfield(B_MATRIX))
    second_feature_subset = get_feature_matrix_hopfield(
        get_weight_matrix_hopfield(I_MATRIX), get_weight_matrix_hopfield(U_MATRIX))
    feature_matrix = get_feature_matrix_hopfield(
        first_feature_subset, second_feature_subset)

    d_matrix_neural = associate_hopfield(get_flatten_matrix(D_MATRIX_NOISED), feature_matrix)
    b_matrix_neural = associate_hopfield(get_flatten_matrix(B_MATRIX_NOISED), feature_matrix)
    i_matrix_neural = associate_hopfield(get_flatten_matrix(I_MATRIX_NOISED), feature_matrix)
    u_matrix_neural = associate_hopfield(get_flatten_matrix(U_MATRIX_NOISED), feature_matrix)

    fig_neural = plt.figure()
    plt.title("Hopfield NN")
    plot_letter(fig_neural, d_matrix_neural, [2, 2, 1])
    plot_letter(fig_neural, b_matrix_neural, [2, 2, 2])
    plot_letter(fig_neural, i_matrix_neural, [2, 2, 3])
    plot_letter(fig_neural, u_matrix_neural, [2, 2, 4])


def Bam_NN():
    first_feature_subset = get_feature_matrix_bam(
        get_weight_matrix_bam(D_MATRIX, D_ASCII), get_weight_matrix_bam(B_MATRIX, B_ASCII))
    second_feature_subset = get_feature_matrix_bam(
        get_weight_matrix_bam(I_MATRIX, I_ASCII), get_weight_matrix_bam(U_MATRIX, U_ASCII))
    feature_matrix = get_feature_matrix_bam(
        first_feature_subset, second_feature_subset)
    
    fig_bam = plt.figure()
    plt.title("BAM NN")
    plot_letter(fig_bam, associate_ascii(D_ASCII, feature_matrix), [2, 2, 1])
    plot_letter(fig_bam, associate_ascii(B_ASCII, feature_matrix), [2, 2, 2])
    plot_letter(fig_bam, associate_ascii(I_ASCII, feature_matrix), [2, 2, 3])
    plot_letter(fig_bam, associate_ascii(U_ASCII, feature_matrix), [2, 2, 4])
    print("Sample D ascii to image")
    print(associate_matrix(D_MATRIX, feature_matrix))
    print("Sample B ascii to image")
    print(associate_matrix(B_MATRIX, feature_matrix))
    print("Sample I ascii to image")
    print(associate_matrix(I_MATRIX, feature_matrix))
    print("Sample U ascii to image")
    print(associate_matrix(U_MATRIX, feature_matrix))
  

if __name__ == "__main__":
    Hopfield_NN()
    Bam_NN()
    plt.show()
