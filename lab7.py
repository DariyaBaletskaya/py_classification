import matplotlib.pyplot as plt
from scipy.io import wavfile as wav
from scipy.fftpack import fft
import numpy as np
import math


def calculate_max_abs_iteration(iterations_array):
    abs_iterations = np.array([])

    for value in iterations_array:
        abs_iterations = np.append(abs_iterations, np.abs(value))
    return np.max(abs_iterations)


def normalize_iterations(iterations_array):
    normalized_array = np.array([])
    max_abs_iteration = calculate_max_abs_iteration(iterations_array)

    for value in iterations_array:
        normalized_array = np.append(normalized_array, value / max_abs_iteration)

    return normalized_array


def find_max_in_slices_of_freq(fft, step):
    start = step
    end = start + step
    ar_of_max_of_fft = np.array([])

    while (end <= 2000):
        ar_of_max_of_fft = np.append(ar_of_max_of_fft, [max(fft[int(start):int(end)])])
        start = end
        end += step

    return ar_of_max_of_fft


# def apply_fast_fourier_transform(filename, ax, plot_position, letter):
#     rate, arr = read_wav(filename)
#     arr_sliced = arr[12000:22000]
#     N = len(arr_sliced)
#     freq = [i * rate / N for i in range(10000)]
#     arr_normalised = normalize_iterations(arr_sliced)
#     fourier_transform = fft(arr_normalised)
#     fourier_transform_abs = np.abs(fourier_transform)
#

#
#     return fourier_transform, freq

def apply_fast_fourier_transform(string):
    rateLetter, Letter = wav.read(string)
    Letter = Letter[22000:55000]
    # Letter = Letter[:33000]
    # print_N_Sound(Letter,'N','Sound')
    fft = make_fft(Letter)
    freq = make_freq(Letter, rateLetter, fft)

    ax_position = ax[plot_position]
    ax_position.plot(freq, fourier_transform_abs, 'r')
    ax_position.title.set_text('Буква ' + letter)
    ax_position.set(xlabel='frequency Hz', ylabel='fft')

    return fft, freq


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


def apply_fast_fourier_transform_nonplot(filename):
    rate, arr = wav.read(filename)
    arr_sliced = arr[12000:22000]
    N = len(arr_sliced)
    freq = [i * rate / N for i in range(10000)]
    arr_normalised = normalize_iterations(arr_sliced)
    fourier_transform = fft(arr_normalised)

    return fourier_transform, freq


def len_of_interval(count_el, amount_of_signs):
    return (count_el / amount_of_signs)


def make_one_aray_of_signs_and_normalize_all(signs):
    arr_U = np.array([])
    arr_E = np.array([])
    arr_YA = np.array([])
    total_arr = np.array([])

    arr_U = np.append(arr_U, signs.arr_of_signs_U1)
    arr_U = np.append(arr_U, signs.arr_of_signs_U2)
    arr_U = np.append(arr_U, signs.arr_of_signs_U3)

    arr_E = np.append(arr_E, signs.arr_of_signs_E1)
    arr_E = np.append(arr_E, signs.arr_of_signs_E2)
    arr_E = np.append(arr_E, signs.arr_of_signs_E3)

    arr_YA = np.append(arr_YA, signs.arr_of_signs_YA1)
    arr_YA = np.append(arr_YA, signs.arr_of_signs_YA2)
    arr_YA = np.append(arr_YA, signs.arr_of_signs_YA3)

    total_arr = np.append(total_arr, arr_U)
    total_arr = np.append(total_arr, arr_E)
    total_arr = np.append(total_arr, arr_YA)

    max_val = max(total_arr)
    normalize_all(signs, max_val)
    return max_val


def normalize_all(signs, max_val):
    normalize_signs(signs.arr_of_signs_U1, max_val)
    normalize_signs(signs.arr_of_signs_U2, max_val)
    normalize_signs(signs.arr_of_signs_U3, max_val)

    normalize_signs(signs.arr_of_signs_E1, max_val)
    normalize_signs(signs.arr_of_signs_E2, max_val)
    normalize_signs(signs.arr_of_signs_E3, max_val)

    normalize_signs(signs.arr_of_signs_YA1, max_val)
    normalize_signs(signs.arr_of_signs_YA2, max_val)
    normalize_signs(signs.arr_of_signs_YA3, max_val)


def normalize_signs(arr, max_val):
    for i in range(len(arr)):
        arr[i] = arr[i] / max_val
    return arr


class Signs(object):
    def __init__(self, FTTS, steps_in_freq):
        self.arr_of_signs_U1 = find_max_in_slices_of_freq(FTTS[0], steps_in_freq)
        self.arr_of_signs_U2 = find_max_in_slices_of_freq(FTTS[1], steps_in_freq)
        self.arr_of_signs_U3 = find_max_in_slices_of_freq(FTTS[2], steps_in_freq)

        self.arr_of_signs_E1 = find_max_in_slices_of_freq(FTTS[3], steps_in_freq)
        self.arr_of_signs_E2 = find_max_in_slices_of_freq(FTTS[4], steps_in_freq)
        self.arr_of_signs_E3 = find_max_in_slices_of_freq(FTTS[5], steps_in_freq)

        self.arr_of_signs_YA1 = find_max_in_slices_of_freq(FTTS[6], steps_in_freq)
        self.arr_of_signs_YA2 = find_max_in_slices_of_freq(FTTS[7], steps_in_freq)
        self.arr_of_signs_YA3 = find_max_in_slices_of_freq(FTTS[8], steps_in_freq)


def RadialBasisNN(string_path, signs, steps_in_feq, max_val):
    fft_L, freq_L = apply_fast_fourier_transform(string_path)
    arr_of_letter_signs = normalize_signs(find_max_in_slices_of_freq(fft_L, steps_in_feq), max_val)
    # print(arr_of_signs_for_Letter)

    ei_U1_letter = find_ei(signs.arr_of_signs_U1, arr_of_letter_signs)
    ei_U2_letter = find_ei(signs.arr_of_signs_U2, arr_of_letter_signs)
    ei_U3_letter = find_ei(signs.arr_of_signs_U3, arr_of_letter_signs)
    ei_E1_letter = find_ei(signs.arr_of_signs_E1, arr_of_letter_signs)
    ei_E2_letter = find_ei(signs.arr_of_signs_E2, arr_of_letter_signs)
    ei_E3_letter = find_ei(signs.arr_of_signs_E3, arr_of_letter_signs)
    ei_YA1_letter = find_ei(signs.arr_of_signs_YA1, arr_of_letter_signs)
    ei_YA2_letter = find_ei(signs.arr_of_signs_YA2, arr_of_letter_signs)
    ei_YA3_letter = find_ei(signs.arr_of_signs_YA3, arr_of_letter_signs)

    arr_ei_for_letter = np.array([])
    arr_ei_for_letter = np.append(arr_ei_for_letter, ei_U1_letter)
    arr_ei_for_letter = np.append(arr_ei_for_letter, ei_U2_letter)
    arr_ei_for_letter = np.append(arr_ei_for_letter, ei_U3_letter)
    arr_ei_for_letter = np.append(arr_ei_for_letter, ei_E1_letter)
    arr_ei_for_letter = np.append(arr_ei_for_letter, ei_E2_letter)
    arr_ei_for_letter = np.append(arr_ei_for_letter, ei_E3_letter)
    arr_ei_for_letter = np.append(arr_ei_for_letter, ei_YA1_letter)
    arr_ei_for_letter = np.append(arr_ei_for_letter, ei_YA2_letter)
    arr_ei_for_letter = np.append(arr_ei_for_letter, ei_YA3_letter)

    max_ei_letter = np.max(arr_ei_for_letter)

    sum_ei_U = ei_U1_letter + ei_U2_letter + ei_U3_letter

    sum_ei_E = ei_E1_letter + ei_E2_letter + ei_E3_letter

    sum_ei_YA = ei_YA1_letter + ei_YA2_letter + ei_YA3_letter

    arr_sum_for_letter = np.array([[sum_ei_U, 'I'], [sum_ei_E, 'E'], [sum_ei_YA, 'O']])
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
        f_ei += (letter_not_identified[i] - letter_identified[i]) ** 2

    ei = math.exp(-(f_ei / 0.02))
    # print(ei)
    return ei


if __name__ == "__main__":
    FILENAMES = ['resources/vowels/I/I1.wav', 'resources/vowels/e/E1.wav', 'resources/vowels/O/O1.wav',
                 'resources/vowels/I/I2.wav', 'resources/vowels/e/E2.wav', 'resources/vowels/O/O2.wav',
                 'resources/vowels/I/I3.wav', 'resources/vowels/e/E3.wav', 'resources/vowels/O/O3.wav']
    LETTERS = ['У', 'Є', 'Я']

    fig, ax = plt.subplots(9)
    fig.set_size_inches(18.5, 10.5)
    fig.subplots_adjust(hspace=.5)
    #
    # feature_U = np.array([])
    # feature_E = np.array([])
    # feature_YA = np.array([])

    fft_U1, freq_U1 = apply_fast_fourier_transform(FILENAMES[0])
    fft_U2, freq_U2 = apply_fast_fourier_transform(FILENAMES[3])
    fft_U3, freq_U3 = apply_fast_fourier_transform(FILENAMES[6])

    fft_E1, freq_E1 = apply_fast_fourier_transform(FILENAMES[1])
    fft_E2, freq_E2 = apply_fast_fourier_transform(FILENAMES[4])
    fft_E3, freq_E3 = apply_fast_fourier_transform(FILENAMES[7])

    fft_YA1, freq_YA1 = apply_fast_fourier_transform(FILENAMES[2])
    fft_YA2, freq_YA2 = apply_fast_fourier_transform(FILENAMES[5])
    fft_YA3, freq_YA3 = apply_fast_fourier_transform(FILENAMES[8])

    plt.show()

    FTTS = [fft_U1, fft_U2, fft_U3,
            fft_E1, fft_E2, fft_E3,
            fft_YA1, fft_YA2, fft_YA3]

    freq_steps = len_of_interval(len(freq_U1), 16)

    signs = Signs(FTTS, freq_steps)

    max_val = make_one_aray_of_signs_and_normalize_all(signs)

    print(FILENAMES[0] + " : " + RadialBasisNN(FILENAMES[0], signs, freq_steps, max_val))
    print(FILENAMES[4] + " : " + RadialBasisNN(FILENAMES[4], signs, freq_steps, max_val))
    print(FILENAMES[7] + " : " + RadialBasisNN(FILENAMES[7], signs, freq_steps, max_val))

    print(FILENAMES[1] + " : " + RadialBasisNN(FILENAMES[1], signs, freq_steps, max_val))
    print(FILENAMES[3] + " : " + RadialBasisNN(FILENAMES[3], signs, freq_steps, max_val))
    print(FILENAMES[6] + " : " + RadialBasisNN(FILENAMES[6], signs, freq_steps, max_val))

    print(FILENAMES[2] + " : " + RadialBasisNN(FILENAMES[2], signs, freq_steps, max_val))
    print(FILENAMES[5] + " : " + RadialBasisNN(FILENAMES[5], signs, freq_steps, max_val))
    print(FILENAMES[8] + " : " + RadialBasisNN(FILENAMES[8], signs, freq_steps, max_val))
