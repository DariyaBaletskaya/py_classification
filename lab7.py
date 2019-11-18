

import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read as s_wav_read
from scipy.fftpack import fft


def read_wav(filename):
    return s_wav_read(filename)


def calculate_max_abs_iteration(iterations_array):
    abs_iterations = np.array([])

    for value in iterations_array:
        abs_iterations = np.append(abs_iterations, np.abs(value))
    return np.max(abs_iterations)


def normalize_iterations(iterations_array):
    normalized_array = np.array([])
    max_abs_iteration = calculate_max_abs_iteration(iterations_array)

    for value in iterations_array:
        normalized_array = np.append(normalized_array, value/max_abs_iteration)

    return(normalized_array)


if __name__ == "__main__":
    rate, arr = read_wav('u.wav')
    arr_sliced = arr[10000:20000]
    N = len(arr_sliced)
    freq = [i*rate/N for i in range(10000)]
    arr_normalised = normalize_iterations(arr_sliced)
    fourier_transform = fft(arr_normalised)
    fourier_transform_abs = np.abs(fourier_transform)

    fig, ax = plt.subplots(2, 2)
    ax[0, 0].plot(freq, fourier_transform, 'r')
    ax[0, 0].title.set_text('Буква У')
    ax[0, 0].set(xlabel='frequency Hz', ylabel='fft')
    plt.show()
