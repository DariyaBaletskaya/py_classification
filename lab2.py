import math as math
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
from Tkinter import *


BUTTON_ENABLE = 1

# ------DATA-----#


class Data:
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
    classes = [class1, class3, class4, class6]
    colors = ['*r', '*g', '*b', '*y']
    naming = ["Class #1", "Class #2", "Class #3", "Class #4"]


class Calculations:
    @staticmethod
    def euclid_distance(class1, class2):
        dx = (class1[0] - class2[0]) ** 2
        dy = (class1[1] - class2[1]) ** 2
        d_euclid = math.sqrt(dx + dy)
        return d_euclid

    # 1.5
    @staticmethod
    def max_mod_distance(class1, class2):
        dx = math.fabs(class1[0] - class2[0])
        dy = math.fabs(class1[1] - class2[1])
        return max([dx, dy])

    # 2.1
    @staticmethod
    def calculate_distance_to_centroid(func, e_point, centroid):
        d = func(e_point, centroid)
        return d

    # 2.1 Util
    @staticmethod
    def find_centroid(eClass):
        x_mean = np.mean(eClass[:, 0])
        y_mean = np.mean(eClass[:, 1])
        return [x_mean, y_mean]

    # 2.5
    @staticmethod
    def calculate_two_minimal_distances(func, e_point, eClass):
        distances = [np.array([])]
        for point in eClass:
            new_distance = func(e_point, point)
            distances = np.append(distances, new_distance)
        return sum(np.sort(distances)[:2])

    # Util
    @staticmethod
    def get_coordinates_between_centroids(first_centroid, second_centroid):
        dx = first_centroid[0] - second_centroid[0]
        dy = first_centroid[1] - second_centroid[1]
        return[dx, dy]


class UI:
    def __init__(self, window):
        self.window = window
        self.fig = Figure(figsize=(6, 6))
        self.plt = self.fig.add_subplot(111)
        self.func1_state = IntVar()
        self.func2_state = IntVar()
        self.func3_state = IntVar()
        self.func4_state = IntVar()
        self.func1 = Checkbutton(
            window, text='(object-centroid) euclid_distance', var=self.func1_state).pack()
        self.func2 = Checkbutton(
            window, text='(object-centroid) max_abs_val_distance', var=self.func2_state).pack()
        self.func3 = Checkbutton(
            window, text='(object-object) euclid_distance', var=self.func3_state).pack()
        self.func4 = Checkbutton(
            window, text='(object-object) max_abs_val_distance', var=self.func4_state).pack()

        self.plot()
        self.plot_centroids()

    def get_color(self, x, y):
        global result
        if x is not None and y is not None:
            calculations = Calculations()
            distances = [np.array([])]
            if self.func1_state.get() == BUTTON_ENABLE:
                for c in Data.classes:
                    d = calculations.calculate_distance_to_centroid(calculations.euclid_distance,
                                                                    [x, y], calculations.find_centroid(c))
                    distances = np.append(distances, d)
                    result = np.where(distances == min(distances))[0][0]
                print(
                    "Calculating distance from point to centroids (Euclid_distance)...")
                print(distances)
                print(Data.naming)
                print("New point goes to" + Data.naming[result])
                Data.classes[result] = np.append(
                    Data.classes[result], [[x, y]], axis=0)
                return Data.colors[result]

            elif self.func2_state.get() == BUTTON_ENABLE:
                for c in Data.classes:
                    d = calculations.calculate_distance_to_centroid(calculations.max_mod_distance,
                                                                    [x, y], calculations.find_centroid(c))
                    distances = np.append(distances, d)
                    result = np.where(distances == min(distances))[0][0]
                print(
                    "Calculating distance from point to centroids (max_abs_val_distance)...")
                print(distances)
                print(Data.naming)
                print("New point goes to" + Data.naming[result])
                Data.classes[result] = np.append(
                    Data.classes[result], [[x, y]], axis=0)
                return Data.colors[result]

            elif self.func3_state.get() == BUTTON_ENABLE:
                for c in Data.classes:
                    d = calculations.calculate_two_minimal_distances(calculations.euclid_distance,
                                                                     [x, y], c)
                    distances = np.append(distances, d)
                    result = np.where(distances == min(distances))[0][0]
                print("Calculating distance from point to objects (Euclid_distance)...")
                print(distances)
                print(Data.naming)
                print("New point goes to" + Data.naming[result])
                Data.classes[result] = np.append(
                    Data.classes[result], [[x, y]], axis=0)
                return Data.colors[result]

            elif self.func4_state.get() == BUTTON_ENABLE:
                for c in Data.classes:
                    d = calculations.calculate_two_minimal_distances(calculations.max_mod_distance,
                                                                     [x, y], c)
                    distances = np.append(distances, d)
                    result = np.where(distances == min(distances))[0][0]
                print(
                    "Calculating distance from point to points (max_abs_val_distance)...")
                print(distances)
                print(Data.naming)
                print("New point goes to" + Data.naming[result])
                Data.classes[result] = np.append(
                    Data.classes[result], [[x, y]], axis=0)
                return Data.colors[result]

            else:
                return ''

    def plot(self):
        for i in enumerate(Data.class1):
            p1, = self.plt.plot(Data.class1[:, 0], Data.class1[:, 1], '*r')
            p2, = self.plt.plot(Data.class3[:, 0], Data.class3[:, 1], '*g')
            p3, = self.plt.plot(Data.class4[:, 0], Data.class4[:, 1], '*b')
            p4, = self.plt.plot(Data.class6[:, 0], Data.class6[:, 1], '*y')

        self.plt.legend([p1, p2, p3, p4], [
                        "class1", "class3", "class4", "class6"])
        self.plt.grid(True)

        def onclick(event):
            self.plt.plot(event.xdata, event.ydata,
                          self.get_color(event.xdata, event.ydata))
            self.fig.canvas.draw()

        canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        canvas.get_tk_widget().pack()
        canvas.draw()
        canvas.mpl_connect('button_press_event', onclick)

    def plot_centroids(self):
        centroids = np.array([[]])
        for c in Data.classes:
            centroid = Calculations.find_centroid(c)
            print(centroid)
            dx = centroid[0]
            dy = centroid[1]
            np.append(centroids, centroid)
            result = np.where(Data.classes == c)[0][0]
            centroid_color = Data.colors[result][1]
            self.plt.plot(dx, dy, "+"+centroid_color)
            print(centroids)


window = Tk()
start = UI(window)
window.mainloop()
