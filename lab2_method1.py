import math as math
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import *
from itertools import combinations

BUTTON_ENABLE = 1


# ------DATA-----#
class Data(object):
    class1 = np.array([[0.49, 0.89], [0.34, 0.81], [0.36, 0.67], [0.47, 0.49], [0.52, 0.53]])
    class3 = np.array([[0.62, 0.83], [0.79, 0.92], [0.71, 0.92], [0.78, 0.83], [0.87, 0.92]])
    class4 = np.array([[0.55, 0.40], [0.66, 0.32], [0.74, 0.49], [0.89, 0.30], [0.77, 0.02]])
    class6 = np.array([[0.05, 0.15], [0.09, 0.39], [0.13, 0.51], [0.25, 0.34], [0.15, 0.36]])

    classes = [class1, class3, class4, class6]
    colors = ['*r', '*g', '*b', '*y']
    naming = ["Class #1", "Class #2", "Class #3", "Class #4"]
    centroids = np.array([[]])


class Calculations(object):
    @staticmethod
    def euclid_distance(class1, class2):
        dx = (class1[0] - class2[0]) ** 2
        dy = (class1[1] - class2[1]) ** 2
        d_euclid = math.sqrt(dx + dy)
        return d_euclid

    @staticmethod
    def max_mod_distance(class1, class2):
        dx = math.fabs(class1[0] - class2[0])
        dy = math.fabs(class1[1] - class2[1])
        return max([dx, dy])

    @staticmethod
    def calculate_distance_to_centroid(func, e_point, centroid):
        d = func(e_point, centroid)
        return d

    @staticmethod
    def find_centroid(eClass):
        x_mean = np.mean(eClass[:, 0])
        y_mean = np.mean(eClass[:, 1])
        return [x_mean, y_mean]

    @staticmethod
    def calculate_two_minimal_distances(func, e_point, eClass):
        distances = [np.array([])]
        for point in eClass:
            new_distance = func(e_point, point)
            distances = np.append(distances, new_distance)
        return sum(np.sort(distances)[:2])

    # Util
    @staticmethod
    def line_equation(a, b, x):
        return a * x + b

    @staticmethod
    def get_mid_section(first_centroid, second_centroid):
        x_o = (first_centroid[0] + second_centroid[0]) / 2
        y_o = (first_centroid[1] + second_centroid[1]) / 2
        return np.array([x_o, y_o])

    @staticmethod
    def get_separating_line(dx, dy, mid_line_point):
        a = -dx / dy
        b = (mid_line_point[0] * dx + mid_line_point[1] * dy) / dy
        # Generate array in the form of [0.1, 0.11, 0.12, ... 0.44]
        coord_x_array = np.arange(0.01, 1, 0.1)
        return {
            'coord_x': coord_x_array,
            'coord_y': Calculations.line_equation(a, b, coord_x_array),
            'a': a,
            'b': b
        }


class UI(object):
    def __init__(self, window):
        self.window = window
        self.fig = Figure(figsize=(6, 6))
        self.plt = self.fig.add_subplot(111)

        self.func1_state = IntVar()
        self.func2_state = IntVar()
        self.func3_state = IntVar()
        self.func4_state = IntVar()
        self.func1 = Checkbutton(window, text='(object-centroid) euclid_distance', var=self.func1_state).pack()
        self.func2 = Checkbutton(window, text='(object-centroid) max_abs_val_distance', var=self.func2_state).pack()
        self.func3 = Checkbutton(window, text='(object-object) euclid_distance', var=self.func3_state).pack()
        self.func4 = Checkbutton(window, text='(object-object) max_abs_val_distance', var=self.func4_state).pack()

        self.divide_classes(self.get_centroids())
        self.classifyClasses()
        self.classifyMatrix()
        self.plot()

    def get_color(self, x, y):
        if x is not None and y is not None:
            calculations = Calculations()
            distances = [np.array([])]
            if self.func1_state.get() == BUTTON_ENABLE:
                for c in Data.classes:
                    d = calculations.calculate_distance_to_centroid(calculations.euclid_distance,
                                                                    [x, y], calculations.find_centroid(c))
                    distances = np.append(distances, d)
                    result = np.where(distances == min(distances))[0][0]
                print("Calculating distance from point to centroids (Euclid_distance)...")
                print(distances)
                print(Data.naming)
                print("New point goes to" + Data.naming[result])
                Data.classes[result] = np.append(Data.classes[result], [[x, y]], axis=0)
                return Data.colors[result]

            elif self.func2_state.get() == BUTTON_ENABLE:
                for c in Data.classes:
                    d = calculations.calculate_distance_to_centroid(calculations.max_mod_distance,
                                                                    [x, y], calculations.find_centroid(c))
                    distances = np.append(distances, d)
                    result = np.where(distances == min(distances))[0][0]
                print("Calculating distance from point to centroids (max_abs_val_distance)...")
                print(distances)
                print(Data.naming)
                print("New point goes to" + Data.naming[result])
                Data.classes[result] = np.append(Data.classes[result], [[x, y]], axis=0)
                return Data.colors[result]

            elif self.func3_state.get() == BUTTON_ENABLE:
                for c in Data.classes:
                    d = calculations.calculate_two_minimal_distances(calculations.euclid_distance, [x, y], c)
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
                    d = calculations.calculate_two_minimal_distances(
                        calculations.max_mod_distance, [x, y], c)
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
        self.plt.set_xlim([0, 1])
        self.plt.set_ylim([0, 1])

        p1, = self.plt.plot(Data.class1[:, 0], Data.class1[:, 1], '*r')
        p2, = self.plt.plot(Data.class3[:, 0], Data.class3[:, 1], '*g')
        p3, = self.plt.plot(Data.class4[:, 0], Data.class4[:, 1], '*b')
        p4, = self.plt.plot(Data.class6[:, 0], Data.class6[:, 1], '*y')
        self.plt.legend([p1, p2, p3, p4], ["class1", "class3", "class4", "class6"])
        self.plt.grid(True)

        def onclick(event):
            self.plt.plot(event.xdata, event.ydata, self.get_color(event.xdata, event.ydata))
            self.fig.canvas.draw()

        canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        canvas.get_tk_widget().pack()
        canvas.draw()
        canvas.mpl_connect('button_press_event', onclick)

    def get_centroids(self):
        centroids = np.array([])
        for c in Data.classes:
            centroid = Calculations.find_centroid(c)
            centroids = np.append(centroids, centroid)
            color_index = np.where(Data.classes == c)[0][0]
            centroid_color = Data.colors[color_index][1]
            self.plt.plot(centroid[0], centroid[1], "+" + centroid_color)
        centroids = centroids.reshape(4, 2)
        return centroids

    def divide_classes(self, centroids):
        # loop through all centroid combinations

        for combo in combinations(centroids, 2):
            first_centroid = combo[0]
            second_centroid = combo[1]
            dx = first_centroid[0] - second_centroid[0]
            dy = first_centroid[1] - second_centroid[1]
            # connect them
            self.plt.plot([first_centroid[0], second_centroid[0]], [first_centroid[1], second_centroid[1]], '--r')
            mid_point = Calculations.get_mid_section(first_centroid, second_centroid)
            # plot center point
            self.plt.plot(mid_point[0], mid_point[1], 'og')
            separating_line = Calculations.get_separating_line(dx, dy, mid_point)
            # plot perpendicular
            self.plt.plot(separating_line['coord_x'], separating_line['coord_y'], ":k")

        for centroid in centroids:
            Data.centroids = np.append(Data.centroids, [[centroid[0]], [centroid[1]]])

        print(Data.centroids)
    def classifyClasses(self):
        matr = np.array([[]])
        for c in Data.classes:
            i = 0
            count1 = 0
            count2 = 0
            count3 = 0
            count4 = 0

            classification = Classification()
            for p in c:
                if classification.classify(p) == 'class1':
                    count1 += 1
                    self.plt.plot(p[0], p[1], 'sg')
                elif classification.classify(p) == 'class2':
                    count2 += 1
                    self.plt.plot(p[0], p[1], 'Dr')
                elif classification.classify(p) == 'class3':
                    count3 += 1
                    self.plt.plot(p[0], p[1], 'ob')
                elif classification.classify(p) == 'class4':
                    count4 += 1
                    self.plt.plot(p[0], p[1], '^y')
                else:
                    self.plt.plot(p[0], p[1], 'ok')

            matr = np.append(matr, np.array([count2, count1, count3, count4]))
            i += 1

        matr = np.reshape(matr, (4, 4))
        print(matr.astype(int))

    def classifyMatrix(self):
        matrix = [[round(x, 2), round(y, 2)] for x in np.arange(0, 1, 0.01) for y in np.arange(0, 1.00, 0.01)]
        classification = Classification()
        for p in matrix:
            if classification.classify(p) == 'class1':
                self.plt.plot(p[0], p[1], 'og')
            elif classification.classify(p) == 'class2':
                self.plt.plot(p[0], p[1], 'or')
            elif classification.classify(p) == 'class3':
                self.plt.plot(p[0], p[1], 'ob')
            elif classification.classify(p) == 'class4':
                self.plt.plot(p[0], p[1], 'oy')
            else:
                self.plt.plot(p[0], p[1], 'ok')


class Perpendiculars(object):

    def getLines(self):
        center1 = [Data.centroids[0], Data.centroids[1]]
        center2 = [Data.centroids[2], Data.centroids[3]]
        center3 = [Data.centroids[4], Data.centroids[5]]
        center4 = [Data.centroids[6], Data.centroids[7]]

        dx_12 = center2[0] - center1[0]
        dy_12 = center2[1] - center1[1]
        dx_13 = center3[0] - center1[0]
        dy_13 = center3[1] - center1[1]
        dx_14 = center4[0] - center1[0]
        dy_14 = center4[1] - center1[1]
        dx_23 = center3[0] - center2[0]
        dy_23 = center3[1] - center2[1]
        dx_42 = center2[0] - center4[0]
        dy_42 = center2[1] - center4[1]
        dx_43 = center3[0] - center4[0]
        dy_43 = center3[1] - center4[1]

        mid_line_p_12 = Calculations.get_mid_section(center1, center2)
        mid_line_p_13 = Calculations.get_mid_section(center1, center3)
        mid_line_p_14 = Calculations.get_mid_section(center1, center4)
        mid_line_p_23 = Calculations.get_mid_section(center2, center3)
        mid_line_p_42 = Calculations.get_mid_section(center4, center4)
        mid_line_p_43 = Calculations.get_mid_section(center4, center3)

        return {
            "sep_line_12": Calculations.get_separating_line(dx_12, dy_12, mid_line_p_12),
            "sep_line_13": Calculations.get_separating_line(dx_13, dy_13, mid_line_p_13),
            "sep_line_14": Calculations.get_separating_line(dx_14, dy_14, mid_line_p_14),
            "sep_line_23": Calculations.get_separating_line(dx_23, dy_23, mid_line_p_23),
            "sep_line_42": Calculations.get_separating_line(dx_42, dy_42, mid_line_p_42),
            "sep_line_43": Calculations.get_separating_line(dx_43, dy_43, mid_line_p_43)
        }


class Classification(object):

    def __init__(self):
        self.lines = Perpendiculars().getLines()

# green
    def is_class1(self, point):
        lines = self.lines
        sep_line_k_12 = [lines['sep_line_12']['a'], lines['sep_line_12']['b']]
        sep_line_k_23 = [lines['sep_line_23']['a'], lines['sep_line_23']['b']]
        # sep_line_k_24 = [-lines['sep_line_42']['a'], -lines['sep_line_42']['b']]
        d12 = point[1] - sep_line_k_12[0] * point[0] - sep_line_k_12[1]
        d23 = point[1] - sep_line_k_23[0] * point[0] - sep_line_k_23[1]
        # d24 = point[1] - sep_line_k_24[0] * point[0] - sep_line_k_24[1]
        return d12 > 0 and d23 > 0

# red
    def is_class2(self, point):
        lines = self.lines
        sep_line_k_14 = [lines['sep_line_14']['a'], lines['sep_line_14']['b']]
        sep_line_k_13 = [lines['sep_line_13']['a'], lines['sep_line_13']['b']]
        sep_line_k_12 = [lines['sep_line_12']['a'], lines['sep_line_12']['b']]
        d14 = point[1] - sep_line_k_14[0] * point[0] - sep_line_k_14[1]
        d13 = point[1] - sep_line_k_13[0] * point[0] - sep_line_k_13[1]
        d12 = point[1] - sep_line_k_12[0] * point[0] - sep_line_k_12[1]
        return d14 > 0 and d13 > 0 and d12 < 0

# blue
    def is_class3(self, point):
        lines = self.lines
        sep_line_k_13 = [lines['sep_line_13']['a'], lines['sep_line_13']['b']]
        sep_line_k_23 = [lines['sep_line_23']['a'], lines['sep_line_23']['b']]
        sep_line_k_34 = [lines['sep_line_43']['a'], lines['sep_line_43']['b']]
        d13 = point[1] - sep_line_k_13[0] * point[0] - sep_line_k_13[1]
        d23 = point[1] - sep_line_k_23[0] * point[0] - sep_line_k_23[1]
        d34 = point[1] - sep_line_k_34[0] * point[0] - sep_line_k_34[1]
        return d13 < 0 and d23 < 0 and d34 < 0

# yellow
    def is_class4(self, point):
        lines = self.lines
        sep_line_k_14 = [lines['sep_line_14']['a'], lines['sep_line_14']['b']]
        sep_line_k_34 = [lines['sep_line_43']['a'], lines['sep_line_43']['b']]
        # sep_line_k_23 = [lines['sep_line_23']['a'], lines['sep_line_23']['b']]
        d14 = point[1] - sep_line_k_14[0] * point[0] - sep_line_k_14[1]
        d34 = point[1] - sep_line_k_34[0] * point[0] - sep_line_k_34[1]
        # d23 = point[1] - sep_line_k_23[0] * point[0] - sep_line_k_23[1]
        return d14 < 0 and d34 > 0

    def classify(self, point):
        c1 = self.is_class1(point)
        c2 = self.is_class2(point)
        c3 = self.is_class3(point)
        c4 = self.is_class4(point)
        if c1 and not (c2 or c3 or c4):
            return 'class1'
        elif c2 and not (c1 or c3 or c4):
            return 'class2'
        elif c3 and not (c1 or c2 or c4):
            return 'class3'
        elif c4 and not (c1 or c2 or c3):
            return 'class4'
        else:
            return 'unclassified'


window = Tk()
start = UI(window)
window.mainloop()
