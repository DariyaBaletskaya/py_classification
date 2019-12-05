import numpy as np
import matplotlib.pyplot as plt
import random as rand
import numpy.linalg as lg

# student id - 10
A = -4
# name letters - 3
B = 1
# surname letters - 5
C = -1
# group num
D = 2

arr_S1 = np.array([])
arr_S2 = np.array([])
arr_S3 = np.array([])
arr_t = np.array([])
matrix_A = 0
matrix_B_S1 = 0
matrix_B_S2 = 0
matrix_B_S3 = 0


def find_arr_S(a=0, b=0, c=0, d=0):
    global arr_t
    s = a + b * arr_t + c * (arr_t ** 2) + d * (arr_t ** 3)
    return s


def create_matrix_A():
    global matrix_A
    n = len(arr_t)
    t = sum_t(1)
    t_2 = sum_t(2)
    t_3 = sum_t(3)
    t_4 = sum_t(4)
    t_5 = sum_t(5)
    t_6 = sum_t(6)

    matrix_A = np.array([
        [n, t, t_2, t_3],
        [t, t_2, t_3, t_4],
        [t_2, t_3, t_4, t_5],
        [t_3, t_4, t_5, t_6]
    ])


def create_matrix_B(S):
    sum_S = sum_st(S)
    sum_St = sum_st(S, 1)
    sum_St_2 = sum_st(S, 2)
    sum_St_3 = sum_st(S, 3)

    matrix_B = np.array([
        [sum_S],
        [sum_St],
        [sum_St_2],
        [sum_St_3]
    ])
    return matrix_B


def sum_t(rate):
    global arr_t
    sum_t = 0
    for i in range(len(arr_t)):
        sum_t = sum_t + arr_t[i] * rate
    return sum_t


def sum_st(S, rate=0):
    sum_st = 0
    for i in range(len(arr_t)):
        sum_st = sum_st + S[i] * (arr_t[i] ** rate)
    return sum_st


def find_result_matrix(m_A, m_B, amount_of_params=4):
    my_matrix_A = m_A[0:amount_of_params, 0:amount_of_params]
    my_matrix_B = m_B[0:amount_of_params, 0:amount_of_params]
    my_matrix_A_inv = lg.inv(my_matrix_A)
    result_val = my_matrix_A_inv.dot(my_matrix_B)
    print("************")
    print(result_val)
    print("************")
    return result_val


def find_E(S, a=0, b=0, c=0, d=0):
    arr_E = S - a - b * arr_t - c * (arr_t ** 2) - d * (arr_t ** 3)
    E = 0
    for i in range(len(arr_E)):
        E = E + arr_E[i]
    E = E ** 2
    return E


def find_E_min(E1, E2, E3):
    arr_E = np.array([E1, E2, E3])
    minimum = min(arr_E)
    return minimum


def add_sh(m_S):
    m_S = m_S + (rand.uniform(1, len(m_S)) - 0.5)
    return m_S


def main_s_plus_sh():
    global arr_t, matrix_A, arr_S2, arr_S3, arr_S1

    S1_sh = add_sh(arr_S1)
    S2_sh = add_sh(arr_S2)
    S3_sh = add_sh(arr_S3)

    create_matrix_A()
    matrix_B_S1_sh = create_matrix_B(S1_sh)
    matrix_B_S2_sh = create_matrix_B(S2_sh)
    matrix_B_S3_sh = create_matrix_B(S3_sh)

    matrix_a_b_for_S11 = find_result_matrix(matrix_A, matrix_B_S1_sh, 2)
    matrix_a_b_c_for_S22 = find_result_matrix(matrix_A, matrix_B_S2_sh, 3)
    matrix_a_b_c_d_for_S33 = find_result_matrix(matrix_A, matrix_B_S3_sh)

    new_matrix_S11 = find_arr_S(matrix_a_b_for_S11[0, 0], matrix_a_b_for_S11[1, 0])
    new_matrix_S22 = find_arr_S(matrix_a_b_c_for_S22[0, 0], matrix_a_b_c_for_S22[1, 0], matrix_a_b_c_for_S22[2, 0])
    new_matrix_S33 = find_arr_S(matrix_a_b_c_d_for_S33[0, 0], matrix_a_b_c_d_for_S33[1, 0],
                                matrix_a_b_c_d_for_S33[2, 0], matrix_a_b_c_d_for_S33[3, 0])

    plt.figure()
    plt.plot(arr_t, arr_S1)
    plt.plot(arr_t, arr_S2)
    plt.plot(arr_t, arr_S3)
    plt.show()

    plt.figure()
    plt.plot(arr_t, arr_S1, 'r')
    plt.plot(arr_t, new_matrix_S11, 'g')
    plt.xlabel("t")
    plt.ylabel("S_1*")
    plt.show()

    plt.figure()
    plt.plot(arr_t, arr_S2, 'r')
    plt.plot(arr_t, new_matrix_S22, 'blue')
    plt.xlabel("t")
    plt.ylabel("S_2*")
    plt.show()

    plt.figure()
    plt.plot(arr_t, arr_S3, 'r')
    plt.plot(arr_t, new_matrix_S33, 'black')
    plt.xlabel("t")
    plt.ylabel("S_3*")
    plt.show()

    E11 = find_E(arr_S1, matrix_a_b_for_S11[0, 0], matrix_a_b_for_S11[1, 0])
    print("E min 1*")
    print(E11)
    print("******")

    E22 = find_E(arr_S2, matrix_a_b_c_for_S22[0, 0], matrix_a_b_c_for_S22[1, 0], matrix_a_b_c_for_S22[2, 0])

    print("E min 2*")
    print(E22)
    print("******")

    E33 = find_E(arr_S3, matrix_a_b_c_d_for_S33[0, 0], matrix_a_b_c_d_for_S33[1, 0], matrix_a_b_c_d_for_S33[2, 0],
                 matrix_a_b_c_d_for_S33[3, 0])

    print("E min 3*")
    print(E33)
    print("******")


def main():
    global matrix_B_S1, arr_S1, A, B, C, D, matrix_B_S2, matrix_B_S3
    global arr_t, matrix_A, arr_S2, arr_S3
    arr_t = np.arange(0, 3, 0.001)
    arr_S1 = find_arr_S(A, B)
    arr_S2 = find_arr_S(A, B, C)
    arr_S3 = find_arr_S(A, B, C, D)

    create_matrix_A()
    matrix_B_S1 = create_matrix_B(arr_S1)
    matrix_B_S2 = create_matrix_B(arr_S2)
    matrix_B_S3 = create_matrix_B(arr_S3)

    matrix_a_b_for_S11 = find_result_matrix(matrix_A, matrix_B_S1, 2)
    matrix_a_b_c_for_S12 = find_result_matrix(matrix_A, matrix_B_S1, 3)
    matrix_a_b_c_d_for_S13 = find_result_matrix(matrix_A, matrix_B_S1)

    matrix_a_b_for_S21 = find_result_matrix(matrix_A, matrix_B_S2, 2)
    matrix_a_b_c_for_S22 = find_result_matrix(matrix_A, matrix_B_S2, 3)
    matrix_a_b_c_d_for_S23 = find_result_matrix(matrix_A, matrix_B_S2)

    matrix_a_b_for_S31 = find_result_matrix(matrix_A, matrix_B_S3, 2)
    matrix_a_b_c_for_S32 = find_result_matrix(matrix_A, matrix_B_S3, 3)
    matrix_a_b_c_d_for_S33 = find_result_matrix(matrix_A, matrix_B_S3)

    new_matrix_S11 = find_arr_S(matrix_a_b_for_S11[0, 0], matrix_a_b_for_S11[1, 0])
    new_matrix_S12 = find_arr_S(matrix_a_b_c_for_S12[0, 0], matrix_a_b_c_for_S12[1, 0], matrix_a_b_c_for_S12[2, 0])
    new_matrix_S13 = find_arr_S(matrix_a_b_c_d_for_S13[0, 0], matrix_a_b_c_d_for_S13[1, 0],
                                matrix_a_b_c_d_for_S13[2, 0], matrix_a_b_c_d_for_S13[3, 0])

    new_matrix_S21 = find_arr_S(matrix_a_b_for_S21[0, 0], matrix_a_b_for_S21[1, 0])
    new_matrix_S22 = find_arr_S(matrix_a_b_c_for_S22[0, 0], matrix_a_b_c_for_S22[1, 0], matrix_a_b_c_for_S22[2, 0])
    new_matrix_S23 = find_arr_S(matrix_a_b_c_d_for_S23[0, 0], matrix_a_b_c_d_for_S23[1, 0],
                                matrix_a_b_c_d_for_S23[2, 0], matrix_a_b_c_d_for_S23[3, 0])

    new_matrix_S31 = find_arr_S(matrix_a_b_for_S31[0, 0], matrix_a_b_for_S31[1, 0])
    new_matrix_S32 = find_arr_S(matrix_a_b_c_for_S32[0, 0], matrix_a_b_c_for_S32[1, 0], matrix_a_b_c_for_S32[2, 0])
    new_matrix_S33 = find_arr_S(matrix_a_b_c_d_for_S33[0, 0], matrix_a_b_c_d_for_S33[1, 0],
                                matrix_a_b_c_d_for_S33[2, 0], matrix_a_b_c_d_for_S33[3, 0])

    plt.figure()
    plt.plot(arr_t, arr_S1)
    plt.plot(arr_t, arr_S2)
    plt.plot(arr_t, arr_S3)
    plt.show()

    plt.figure()
    plt.plot(arr_t, arr_S1, 'r')
    plt.plot(arr_t, new_matrix_S11, 'g')
    plt.plot(arr_t, new_matrix_S12, 'blue')
    plt.xlabel("t")
    plt.ylabel("S_1*_sh")
    plt.show()

    plt.plot(arr_t, new_matrix_S13, 'black')
    plt.xlabel("t")
    plt.ylabel("S_1*_sh")
    plt.show()

    plt.figure()
    plt.plot(arr_t, arr_S2, 'r')
    plt.plot(arr_t, new_matrix_S21, 'g')
    plt.plot(arr_t, new_matrix_S22, 'blue')
    plt.xlabel("t")
    plt.ylabel("S_2*_sh")
    plt.show()

    plt.plot(arr_t, new_matrix_S23, 'black')
    plt.xlabel("t")
    plt.ylabel("S_2*_sh")
    plt.show()

    plt.figure()
    plt.plot(arr_t, arr_S3, 'r')
    plt.plot(arr_t, new_matrix_S31, 'g')
    plt.plot(arr_t, new_matrix_S32, 'blue')
    plt.xlabel("t")
    plt.ylabel("S_3*_sh")
    plt.show()

    plt.plot(arr_t, new_matrix_S33, 'black')
    plt.xlabel("t")
    plt.ylabel("S_3*_sh")
    plt.show()

    E11 = find_E(arr_S1, matrix_a_b_for_S11[0, 0], matrix_a_b_for_S11[1, 0])
    E12 = find_E(arr_S1, matrix_a_b_c_for_S12[0, 0], matrix_a_b_c_for_S12[1, 0], matrix_a_b_c_for_S12[2, 0])
    E13 = find_E(arr_S1, matrix_a_b_c_d_for_S13[0, 0], matrix_a_b_c_d_for_S13[1, 0], matrix_a_b_c_d_for_S13[2, 0],
                 matrix_a_b_c_d_for_S13[3, 0])

    print("E min 1*")
    print(E11)
    print(E12)
    print(E13)
    print("******")
    print(find_E_min(E11, E12, E13))
    print()

    E21 = find_E(arr_S2, matrix_a_b_for_S21[0, 0], matrix_a_b_for_S21[1, 0])
    E22 = find_E(arr_S2, matrix_a_b_c_for_S22[0, 0], matrix_a_b_c_for_S22[1, 0], matrix_a_b_c_for_S22[2, 0])
    E23 = find_E(arr_S2, matrix_a_b_c_d_for_S23[0, 0], matrix_a_b_c_d_for_S23[1, 0], matrix_a_b_c_d_for_S23[2, 0],
                 matrix_a_b_c_d_for_S23[3, 0])

    print("E min 2*")
    print(E21)
    print(E22)
    print(E23)
    print("******")
    print(find_E_min(E21, E22, E23))
    print()

    E31 = find_E(arr_S3, matrix_a_b_for_S31[0, 0], matrix_a_b_for_S31[1, 0])
    E32 = find_E(arr_S3, matrix_a_b_c_for_S32[0, 0], matrix_a_b_c_for_S32[1, 0], matrix_a_b_c_for_S32[2, 0])
    E33 = find_E(arr_S3, matrix_a_b_c_d_for_S33[0, 0], matrix_a_b_c_d_for_S33[1, 0], matrix_a_b_c_d_for_S33[2, 0],
                 matrix_a_b_c_d_for_S33[3, 0])

    print("E min 3*")
    print(E31)
    print(E32)
    print(E33)
    print("******")
    print(find_E_min(E31, E32, E33))
    print()

    main_s_plus_sh()


if __name__ == "__main__":
    main()
