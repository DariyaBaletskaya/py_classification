

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

    return normalized_array


def apply_fast_fourier_transform(filename, ax, plot_position, letter):
    rate, arr = read_wav(filename)
    arr_sliced = arr[10000:20000]
    N = len(arr_sliced)
    freq = [i*rate/N for i in range(10000)]
    arr_normalised = normalize_iterations(arr_sliced)
    fourier_transform = fft(arr_normalised)
    fourier_transform_abs = np.abs(fourier_transform)

    ax_position = ax[plot_position]
    ax_position.plot(freq, fourier_transform_abs, 'r')
    ax_position.title.set_text('Буква ' + letter)
    ax_position.set(xlabel='frequency Hz', ylabel='fft')


if __name__ == "__main__":

    FILENAMES = ['u.wav', 'e.wav', 'a.wav']
    LETTERS = ['У', 'Є', 'Я']
    
    fig, ax = plt.subplots(3)
    fig.set_size_inches(18.5, 10.5)
    fig.subplots_adjust(hspace=.5)
    
    for index, file in enumerate(FILENAMES):
        apply_fast_fourier_transform(file, ax, index, LETTERS[index])
    plt.show()
