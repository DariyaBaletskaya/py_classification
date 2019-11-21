import numpy as np
import math
from tkinter import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


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
    def radial_basis(classes, point):
        class_activities = np.array([])
        for i, class_i in enumerate(classes):
            class_activity = np.array([])
            for el in class_i:
                act = Calculations.activity_func(el, point)
                class_activity = np.append(class_activity, act)

            class_activities = np.append(class_activities, class_activity)
            class_activities = np.reshape(class_activities, (i + 1, -1))

        total_class_activities = np.array([])
        for activity in class_activities:
            total_class_activities = np.append(
                total_class_activities, sum(activity))

        max_activity_index = np.where(
            total_class_activities == np.amax(total_class_activities))[0]
        # print("class activities:\n{0}".format(class_activities))
        print("total class activities:\n{0}".format(total_class_activities))
        print("x: {0}, y: {1}, class: {2}".format(
            point[0], point[1], max_activity_index))

        return max_activity_index

    @staticmethod
    def activity_func(el, point):
        return math.exp(-((el[0] - point[0]) ** 2 + (el[1] - point[1]) ** 2) / 0.1 ** 2)


class UI:
    def __init__(self, window):
        self.window = window
        self.fig = Figure(figsize=(6, 6))
        self.plt = self.fig.add_subplot(111)
        self.plot()

    def plot(self):
        points = [[round(x, 2), round(y, 2)] for x in np.arange(
            0, 1.01, 0.1) for y in np.arange(0, 1.01, 0.1)]
        self.draw_classes()
        classes = [Data.class1, Data.class3, Data.class4, Data.class6]
        for p in points:
            point_class = Calculations.radial_basis(classes, p)
            self.draw_point(p, point_class)

        def onclick(event):
            point = [event.xdata, event.ydata]
            self.draw_point(point, Calculations.radial_basis(classes, point))
            self.fig.canvas.draw()

        canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        canvas.get_tk_widget().pack()
        canvas.draw()
        canvas.mpl_connect('button_press_event', onclick)

    def draw_classes(self):
        p1, = self.plt.plot(Data.class1[:, 0], Data.class1[:, 1], 'or')
        p2, = self.plt.plot(Data.class3[:, 0], Data.class3[:, 1], 'ob')
        p5, = self.plt.plot(Data.class4[:, 0], Data.class4[:, 1], 'ok')
        p6, = self.plt.plot(Data.class6[:, 0], Data.class6[:, 1], 'og')

        self.plt.legend([p1, p2, p5, p6], [
                        "class1", "class2", "class3", "class4"])
        self.plt.grid(True)

    def draw_point(self, point, class_i):
        if class_i == 0:
            color = 'red'
        if class_i == 1:
            color = 'blue'
        if class_i == 2:
            color = "black"
        if class_i == 3:
            color = "green"

        self.plt.plot(point[0], point[1], marker='*', color=color)


def main():
    window = Tk()
    UI(window)
    window.mainloop()


if __name__ == '__main__':
    main()
