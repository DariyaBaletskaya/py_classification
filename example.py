# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

class_red = np.array([[0.05, 0.91],
                      [0.14, 0.96],
                      [0.16, 0.9],
                      [0.07, 0.7],
                      [0.2, 0.63]])

class_blue = np.array([[0.49, 0.89],
                       [0.34, 0.81],
                       [0.36, 0.67],
                       [0.47, 0.49],
                       [0.52, 0.53]])


def line_equation(a, b, x):
    """
      Функция уравнения прямой 
    """
    return a * x + b


def get_center_class_red(some_class):
    return np.array([0.125, 0.795])


def get_center_class_blue(some_class):
    return np.array([0.43, 0.69])


def show_line_equation_connect_classes(dx, dy, center_class_red, center_class_blue):
    """
      Определение уравнения прямой, соединяющей центроиды
    """
    a1 = dy / dx
    b1 = center_class_red[1] - a1 * center_class_red[0]
    x1 = np.arange(center_class_red[0], center_class_blue[0], 0.05)
    y1 = line_equation(a1, b1, x1)
    return x1, y1


def get_mid_section(coord_x, coord_y):
    """
      Середина отрезка, соединяющего центроиды
    """
    x_o = coord_x + abs(dx) / 2
    y_o = coord_y + abs(dy) / 2
    return np.array([x_o, y_o])  # p_o - середина отрезка точка О


def get_separating_line(dx, dy, min_line_point):
    """
      Определение уравнения прямой, проходящей через точку O 
      перпендикулярно прямой, соединяющей центроиды&
    """
    a = -dx / dy
    b = (min_line_point[0] * dx + min_line_point[1] * dy) / dy
    # Генерируем массив в виде [0.1, 0.11, 0.12, ... 0.44]
    coord_x_array = np.arange(0.1, 0.45, 0.01)
    return {
        'coord_x': coord_x_array,
        'coord_y': line_equation(a, b, coord_x_array),
    }


# Берем центры классов
center_class_red = get_center_class_red(class_red)
center_class_blue = get_center_class_blue(class_blue)

# Вычисляем координаты
dx = center_class_blue[0] - center_class_red[0]
dy = center_class_blue[1] - center_class_red[1]

# Рисуем центры классов
plt.plot(center_class_red[0], center_class_red[1], '+r')
plt.plot(center_class_blue[0], center_class_blue[1], '+b')


# Вычисляем среднюю точку между классами
min_line_point = get_mid_section(center_class_red[0], center_class_blue[1])
# Рисуем точку на середине отрезка между классами.
plt.plot(min_line_point[0], min_line_point[1], 'og')

# Вычисляем линию проходящую между классами
line_x, line_y = show_line_equation_connect_classes(
    dx, dy, center_class_red, center_class_blue)
plt.plot(line_x, line_y)

# Вычисляем линию, разделяющую классы
separating_line = get_separating_line(dx, dy, min_line_point)
# Рисиуем линию, разделяющую классы
plt.plot(separating_line['coord_x'], separating_line['coord_y'])


p1 = plt.plot(class_red[:, 0], class_red[:, 1], '*r')
p2 = plt.plot(class_blue[:, 0], class_blue[:, 1], 'sb')
plt.legend([p1, p2], ["Class red", "Class blue"])
plt.grid(True)
plt.show()
