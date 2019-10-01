'''
ЛАБ 2 - ЧАСТЬ 2 
Пример для этапов 7-8

'''
##1
import numpy as np
import matplotlib.pyplot as plt


# Функция уравнения прямой y=a*x+b
def straight_line_equation(a, b, x):
    return a * x + b


# уравнение для решающей функции  dij (x , y )= c*y - b*x-a
def dij_method(c, a, b, point):
    return c * point[1] - a * point[0] - b


# Класс 1
class1 = np.array([[0.05, 0.91],
                   [0.14, 0.96],
                   [0.16, 0.9],
                   [0.07, 0.7],
                   [0.2, 0.63]])
# Класс 2
class2 = np.array([[0.49, 0.89],
                   [0.34, 0.81],
                   [0.36, 0.67],
                   [0.47, 0.49],
                   [0.52, 0.53]])
# Класс 3
class3 = np.array([[0.55, 0.4],
                   [0.66, 0.32],
                   [0.74, 0.49],
                   [0.89, 0.3],
                   [0.77, 0.2]])
class6 = np.array([[0.05, 0.15], [0.09, 0.39], [0.13, 0.51], [0.25, 0.34], [0.15, 0.36]])

# Центроиды класса 1 - centroid_сlass1, класса 2 - centroid_сlass2, класса 3 - centroid_сlass3
centroid_сlass1 = np.array([0.125, 0.795])
centroid_сlass2 = np.array([0.43, 0.69])
centroid_сlass3 = np.array([0.72, 0.345])
centroid_сlass4 = np.array([0.22, 0.54])

a12 = 2.9047
b12 = -0.0636
x12 = np.arange(0, 0.38, 0.01)
y12 = straight_line_equation(a12, b12, x12)
plt.plot(x12, y12)

a13 = 1.3222
b13 = 0.01136
x13 = np.arange(0.1, 0.75, 0.01)
y13 = straight_line_equation(a13, b13, x13)

a23 = 0.84058
b23 = 0.03417
x23 = np.arange(0.058, 1, 0.01)
y23 = straight_line_equation(a23, b23, x23)

# # Коэффициенты уравнения границ между классами c*y-a*x-b=0
params_class1 = np.array([[1, a12, b12], [1, a13, b13]])
params_class2 = np.array([[1, a12, b12], [1, a23, b23]])
params_class3 = np.array([[1, a13, b13], [1, a23, b23]])

## для класса 1
d_C1y12 = dij_method(1, a12, b12, centroid_сlass1)
d_C1y13 = dij_method(1, a13, b13, centroid_сlass1)

if d_C1y12 < 0:
    params_class1[0, :] = - params_class1[0, :]
if d_C1y13 < 0:
    params_class1[1, :] = - params_class1[1, :]

# для класса 2
d_C2y12 = dij_method(1, a12, b12, centroid_сlass2)
d_C2y23 = dij_method(1, a23, b23, centroid_сlass2)
if d_C2y12 < 0:
    params_class2[0, :] = - params_class2[0, :]
if d_C2y23 < 0:
    params_class2[1, :] = - params_class2[1, :]

# для класса 3
d_C3y13 = dij_method(1, a13, b13, centroid_сlass3)
d_C3y23 = dij_method(1, a23, b23, centroid_сlass3)
if d_C3y13 < 0:
    params_class3[0, :] = - params_class3[0, :]
if d_C3y23 < 0:
    params_class3[1, :] = - params_class3[1, :]

# для последовательного разделения классов
x = np.arange(0, 1, 0.1)
y = np.arange(0, 1, 0.1)
xy = np.zeros((len(x) * len(y), 2))
ni = 0
for i in range(len(x)):
    for j in range(len(y)):
        ni += 1
        xy[ni:] = [x[i], y[j]]
points = xy

new_cl1 = class1
new_cl2 = class2
new_cl3 = class3
[nr, nc] = points.shape
for i in range(nr):
    p = points[i, :]
    a = params_class1[0, 1]
    b = params_class1[0, 2]
    c = params_class1[0, 0]

    if dij_method(c, a, b, p) > 0:
        new_cl1 = np.vstack([new_cl1, p])
    else:
        a = params_class2[1, 1]
        b = params_class2[1, 2]
        c = params_class2[1, 0]

        if dij_method(c, a, b, p) > 0:
            new_cl2 = np.vstack([new_cl2, p])
        else:
            new_cl3 = np.vstack([new_cl3, p])

## Построение рисунка
p1 = plt.plot(new_cl1[:, 0], new_cl1[:, 1], '*r')
p2 = plt.plot(new_cl2[:, 0], new_cl2[:, 1], 'sb')
p3 = plt.plot(new_cl3[:, 0], new_cl3[:, 1], 'om')
plt.plot(x12, y12)
plt.plot(x23, y23)

plt.grid(True)
plt.show()
