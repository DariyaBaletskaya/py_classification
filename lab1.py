import math as math
import numpy as np
import matplotlib.pyplot as plt

# classes of etalon values
class1 = np.array([[0.49, 0.89],
                   [0.34, 0.81],
                   [0.36, 0.67],
                   [0.47, 0.49],
                   [0.52, 0.53]])

class3 = np.array([[0.62, 0.83],
                   [0.79, 0.92],
                   [0.71, 0.92],
                   [0.78, 0.83],
                   [0.87, 0.92]])

class4 = np.array([[0.55, 0.4],
                   [0.66, 0.32],
                   [0.74, 0.49],
                   [0.89, 0.3],
                   [0.77, 0.02]])


class6 = np.array([[0.05, 0.15],
                   [0.09, 0.39],
                   [0.13, 0.51],
                   [0.25, 0.34],
                   [0.15, 0.36]])

# 1.1


def euclid_distance(class1, class2):
    dx = (class1[0]-class2[0])**2
    dy = (class2[1]-class2[1])**2
    d_euclid = math.sqrt(dx+dy)
    return d_euclid

# 1.5


def max_mod_distance(class1, class2):
    dx = math.fabs(class1[0]-class2[0])
    dy = math.fabs(class2[1]-class2[1])
    return max([dx, dy])

# 2.1


def find_centroid(eClass):
    xMean = np.mean(eClass[:, 0])
    yMean = np.mean(eClass[:, 1])
    return [xMean, yMean]


def calculate_distance_to_centroid(func, ePoint, centroid):
    d = func(ePoint, centroid)
    return d


def calculate_two_minimal_distances(func, ePoint, eClass):
    distances = [np.array([])]
    for point in eClass:
        newDistance = func(ePoint, point)
        distances = np.append(distances, newDistance)
    return sum(np.sort(distances)[:2]


for i in enumerate(class1):
    p1, = plt.plot(class1[:, 0], class1[:, 1], '*r')

for i in enumerate(class3):
    p2, = plt.plot(class3[:, 0], class3[:, 1], '*g')

for i in enumerate(class4):
    p3, = plt.plot(class4[:, 0], class4[:, 1], '*b')

for i in enumerate(class6):
    p4, = plt.plot(class6[:, 0], class6[:, 1], '*y')

plt.legend([p1, p2, p3, p4], ["class1", "class3", "class4", "class6"])
plt.grid(True)
plt.show()
