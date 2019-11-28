# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 12:15:42 2019

@author: Admin
"""
import matplotlib.pyplot as plt
from scipy.io import wavfile as wav
from scipy.fftpack import fft
import numpy as np
import math


arr_of_signs_A1 = 0
arr_of_signs_A2 = 0
arr_of_signs_A3 = 0
arr_of_signs_O2 = 0
arr_of_signs_O3 = 0
arr_of_signs_O1 = 0
arr_of_signs_I1 = 0
arr_of_signs_I2 = 0
arr_of_signs_I3 = 0
max_val = 0
steps_in_freq = 0


arr_of_A = np.array([])
arr_of_O = np.array([])
arr_of_I = np.array([])

arr_of_all_signs = np.array([])


def print_N_Sound(data, xlabel, ylabel):
    # 1
    plt.figure()
    plt.plot(data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def make_fft(data2):
    fft_out = np.abs(fft(data2))
    # print_N_Sound(fft_out,'N','fft')
    return fft_out


def make_freq(data3, rate, fft_out):
    # 3
    N = 28000
    freq = [i * rate / N for i in range(2000)]
    # plt.plot(freq,fft_out[0:2000])
    # plt.xlabel('frequency Hz')
    # plt.ylabel('fft')
    # plt.show()
    return freq


def len_of_interval(count_el, amount_of_signs):
    return (count_el / amount_of_signs)


def find_max_in_slices_of_freq(fft, step):
    start = step
    end = start + step
    ar_of_max_of_fft = np.array([])

    while (end <= 2000):
        ar_of_max_of_fft = np.append(ar_of_max_of_fft, [max(fft[int(start):int(end)])])
        start = end
        end += step

    return ar_of_max_of_fft


def get_fft_and_freq(string):
    rateLetter, Letter = wav.read(string)
    Letter = Letter[22000:55000]
    Letter = Letter[:, 0]
    # print_N_Sound(Letter,'N','Sound')
    fft = make_fft(Letter)
    freq = make_freq(Letter, rateLetter, fft)
    return fft, freq


def normalize_signs(arr, max_val):
    for i in range(len(arr)):
        arr[i] = arr[i] / max_val
    return arr


def normalize_all():
    global arr_of_signs_A1, arr_of_signs_A2, arr_of_signs_A3
    global arr_of_signs_O2, arr_of_signs_O3, arr_of_signs_O1
    global arr_of_signs_I1, arr_of_signs_I2, arr_of_signs_I3
    global max_val

    normalize_signs(arr_of_signs_A1, max_val)
    normalize_signs(arr_of_signs_A2, max_val)
    normalize_signs(arr_of_signs_A3, max_val)

    normalize_signs(arr_of_signs_O1, max_val)
    normalize_signs(arr_of_signs_O2, max_val)
    normalize_signs(arr_of_signs_O3, max_val)

    normalize_signs(arr_of_signs_I1, max_val)
    normalize_signs(arr_of_signs_I2, max_val)
    normalize_signs(arr_of_signs_I3, max_val)


def find_repeatable_elements(arr):
    new_arr = np.array([])
    for i in range(len(arr)):
        elem = arr[i]
        cnt = 0
        for j in range(len(arr)):
            if (arr[j] == elem):
                cnt += 1
        if (cnt == 1):
            new_arr = np.append(new_arr, [elem])
    print(new_arr)
    return arr


def make_one_aray_of_signs_and_normalize_all():
    global arr_of_all_signs, max_val
    global arr_of_signs_A1, arr_of_signs_A2, arr_of_signs_A3
    global arr_of_signs_O2, arr_of_signs_O3, arr_of_signs_O1
    global arr_of_signs_I1, arr_of_signs_I2, arr_of_signs_I3



    arr_A = np.array([])
    arr_O = np.array([])
    arr_I = np.array([])
    total_arr = np.array([])

    arr_A = np.append(arr_A, arr_of_signs_A1)
    arr_A = np.append(arr_A, arr_of_signs_A2)
    arr_A = np.append(arr_A, arr_of_signs_A3)

    arr_O = np.append(arr_O, arr_of_signs_O1)
    arr_O = np.append(arr_O, arr_of_signs_O2)
    arr_O = np.append(arr_O, arr_of_signs_O3)

    arr_I = np.append(arr_I, arr_of_signs_I1)
    arr_I = np.append(arr_I, arr_of_signs_I2)
    arr_I = np.append(arr_I, arr_of_signs_I3)

    total_arr = np.append(total_arr, arr_A)
    total_arr = np.append(total_arr, arr_O)
    total_arr = np.append(total_arr, arr_I)

    max_val = max(total_arr)
    normalize_all()

    global steps_in_freq

    plt.figure()
    plt.plot(arr_of_signs_A1, ".g")
    plt.plot(arr_of_signs_A2, ".g")
    plt.plot(arr_of_signs_A3, ".g")
    plt.plot(arr_of_signs_O1, ".b")
    plt.plot(arr_of_signs_O2, ".b")
    plt.plot(arr_of_signs_O3, ".b")
    plt.plot(arr_of_signs_I1, ".r")
    plt.plot(arr_of_signs_I2, ".r")
    plt.plot(arr_of_signs_I3, ".r")

    plt.xlabel("N")
    plt.ylabel("A")
    plt.show()


def build_neural_network(string_path):
    global steps_in_freq, max_val
    fft_L, freq_L = get_fft_and_freq(string_path)
    arr_of_signs_for_Letter = find_max_in_slices_of_freq(fft_L, steps_in_freq)
    arr_of_signs_for_Letter = normalize_signs(arr_of_signs_for_Letter, max_val)
    # print(arr_of_signs_for_Letter)

    ei_A1_letter = find_ei(arr_of_signs_A1, arr_of_signs_for_Letter)
    ei_A2_letter = find_ei(arr_of_signs_A2, arr_of_signs_for_Letter)
    ei_A3_letter = find_ei(arr_of_signs_A3, arr_of_signs_for_Letter)
    ei_O1_letter = find_ei(arr_of_signs_O1, arr_of_signs_for_Letter)
    ei_O2_letter = find_ei(arr_of_signs_O2, arr_of_signs_for_Letter)
    ei_O3_letter = find_ei(arr_of_signs_O3, arr_of_signs_for_Letter)
    ei_I1_letter = find_ei(arr_of_signs_I1, arr_of_signs_for_Letter)
    ei_I2_letter = find_ei(arr_of_signs_I2, arr_of_signs_for_Letter)
    ei_I3_letter = find_ei(arr_of_signs_I3, arr_of_signs_for_Letter)

    arr_ei_for_letter = np.array([])
    arr_ei_for_letter = np.append(arr_ei_for_letter, ei_A1_letter)
    arr_ei_for_letter = np.append(arr_ei_for_letter, ei_A2_letter)
    arr_ei_for_letter = np.append(arr_ei_for_letter, ei_A3_letter)
    arr_ei_for_letter = np.append(arr_ei_for_letter, ei_O1_letter)
    arr_ei_for_letter = np.append(arr_ei_for_letter, ei_O2_letter)
    arr_ei_for_letter = np.append(arr_ei_for_letter, ei_O3_letter)
    arr_ei_for_letter = np.append(arr_ei_for_letter, ei_I1_letter)
    arr_ei_for_letter = np.append(arr_ei_for_letter, ei_I2_letter)
    arr_ei_for_letter = np.append(arr_ei_for_letter, ei_I3_letter)

    max_ei_letter = np.max(arr_ei_for_letter)

    sum_ei_A = ei_A1_letter + ei_A2_letter + ei_A3_letter

    sum_ei_O = ei_O1_letter + ei_O2_letter + ei_O3_letter

    sum_ei_I = ei_I1_letter + ei_I2_letter + ei_I3_letter

    arr_sum_for_letter = np.array([[sum_ei_A, 'A'], [sum_ei_O, 'O'], [sum_ei_I, 'I']])
    # print(arr_sum_for_letter)
    arr_sum_for_letter = arr_sum_for_letter[arr_sum_for_letter[:, 0].argsort()]
    # print(arr_sum_for_letter)

    # print("++---")
    # print(max_ei_letter)
    # print(max_ei_letter < float(arr_sum_for_letter[2][0]))
    if (max_ei_letter < 0.001):
        return "undefined"
    elif (max_ei_letter <= float(arr_sum_for_letter[0][0])):
        return arr_sum_for_letter[0][1]
    elif (max_ei_letter <= float(arr_sum_for_letter[1][0])):
        return arr_sum_for_letter[1][1]
    elif (max_ei_letter <= float(arr_sum_for_letter[2][0])):
        return arr_sum_for_letter[2][1]



def find_ei(letter_identified, letter_not_identified):
    f_ei = 0
    for i in range(len(letter_identified)):
        f_ei += (letter_not_identified[i] - letter_identified[i])**2


    ei = math.exp(-(f_ei / 0.02))
    # print(ei)
    return ei


def main():
    global arr_of_signs_A1, arr_of_signs_A2, arr_of_signs_A3
    global arr_of_signs_O2, arr_of_signs_O3, arr_of_signs_O1
    global arr_of_signs_I1, arr_of_signs_I2, arr_of_signs_I3
    global arr_of_A, arr_of_O, arr_of_I
    global steps_in_freq

    FILENAMES = ['resources/vowels/u/u_1.wav', 'resources/vowels/e/e_1.wav', 'resources/vowels/a/a_1.wav',
                 'resources/vowels/u/u_2.wav', 'resources/vowels/e/e_2.wav', 'resources/vowels/a/a_2.wav',
                 'resources/vowels/u/u_3.wav', 'resources/vowels/e/e_3.wav', 'resources/vowels/a/a_3.wav']

    fft_A1, freq_A1 = get_fft_and_freq(FILENAMES[0])
    fft_A2, freq_A2 = get_fft_and_freq(FILENAMES[3])
    fft_A3, freq_A3 = get_fft_and_freq(FILENAMES[6])

    fft_O1, freq_O1 = get_fft_and_freq(FILENAMES[1])
    fft_O2, freq_O2 = get_fft_and_freq(FILENAMES[4])
    fft_O3, freq_O3 = get_fft_and_freq(FILENAMES[7])

    fft_I1, freq_I1 = get_fft_and_freq(FILENAMES[2])
    fft_I2, freq_I2 = get_fft_and_freq(FILENAMES[5])
    fft_I3, freq_I3 = get_fft_and_freq(FILENAMES[8])

    steps_in_freq = len_of_interval(len(freq_A1), 16)

    arr_of_signs_A1 = find_max_in_slices_of_freq(fft_A1, steps_in_freq)
    arr_of_signs_A2 = find_max_in_slices_of_freq(fft_A2, steps_in_freq)
    arr_of_signs_A3 = find_max_in_slices_of_freq(fft_A3, steps_in_freq)

    arr_of_signs_O1 = find_max_in_slices_of_freq(fft_O1, steps_in_freq)
    arr_of_signs_O2 = find_max_in_slices_of_freq(fft_O2, steps_in_freq)
    arr_of_signs_O3 = find_max_in_slices_of_freq(fft_O3, steps_in_freq)

    arr_of_signs_I1 = find_max_in_slices_of_freq(fft_I1, steps_in_freq)
    arr_of_signs_I2 = find_max_in_slices_of_freq(fft_I2, steps_in_freq)
    arr_of_signs_I3 = find_max_in_slices_of_freq(fft_I3, steps_in_freq)

    make_one_aray_of_signs_and_normalize_all()

    # print("/Users/citrus/Desktop/lab_7/another_letters/B.wav" + " : " + build_neural_network("/Users/citrus/Desktop/lab_7/another_letters/B.wav"))
    # print("/Users/citrus/Desktop/lab_7/another_letters/P.wav" + " : " + build_neural_network("/Users/citrus/Desktop/lab_7/another_letters/P.wav"))
    print(FILENAMES[0] + " : " + build_neural_network(FILENAMES[0]))
    print(FILENAMES[1] + " : " + build_neural_network(FILENAMES[1]))
    print(FILENAMES[2] + " : " + build_neural_network(FILENAMES[2]))

    print(FILENAMES[3] + " : " + build_neural_network(FILENAMES[3]))
    print(FILENAMES[4] + " : " + build_neural_network(FILENAMES[4]))
    print(FILENAMES[5] + " : " + build_neural_network(FILENAMES[5]))

    print(FILENAMES[6] + " : " + build_neural_network(FILENAMES[6]))
    print(FILENAMES[7] + " : " + build_neural_network(FILENAMES[7]))
    print(FILENAMES[8] + " : " + build_neural_network(FILENAMES[8]))


if __name__ == "__main__":
    main()
