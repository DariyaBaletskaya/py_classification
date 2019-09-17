import math as math
import numpy as np
import matplotlib.pyplot as plt

from tkinter import *
from tkinter.ttk import *

window = Tk()
window.title("Welcome to LikeGeeks app")
window.geometry('350x200')
chk_state = BooleanVar()
chk_state.set(True)
chk = Checkbutton(window, text='Choose', var=chk_state)
chk = Checkbutton(window, text='Choose', var=chk_state)
chk.grid(column=0, row=0)

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
    dx = (class1[0] - class2[0]) ** 2
    dy = (class2[1] - class2[1]) ** 2
    d_euclid = math.sqrt(dx + dy)
    return d_euclid


# 1.5
def max_mod_distance(class1, class2):
    dx = math.fabs(class1[0] - class2[0])
    dy = math.fabs(class2[1] - class2[1])
    return max([dx, dy])


# 2.1
def calculate_distance_to_centroid(func, ePoint, centroid):
    d = func(ePoint, centroid)
    return d


# 2.1 Util
def find_centroid(eClass):
    xMean = np.mean(eClass[:, 0])
    yMean = np.mean(eClass[:, 1])
    return [xMean, yMean]


# 2.5
def calculate_two_minimal_distances(func, ePoint, eClass):
    distances = [np.array([])]
    for point in eClass:
        newDistance = func(ePoint, point)
        distances = np.append(distances, newDistance)
    return sum(np.sort(distances))[:2]


fig = plt.figure()
ax = fig.add_subplot(111)

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


def onclick(event):
    print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          (event.button, event.x, event.y, event.xdata, event.ydata))
    plt.plot(event.xdata, event.ydata, '*r')
    fig.canvas.draw()


cid = fig.canvas.mpl_connect('button_press_event', onclick)
window.mainloop()

plt.show()
