import constants as const
import numpy as np
import matplotlib.pyplot as plt
import math

class1 = np.array([[0.49, 0.89],
                   [0.34, 0.81],
                   [0.36, 0.67],
                   [0.47, 0.49],
                   [0.52, 0.53]])
class2 = np.array([[0.62, 0.83],
                   [0.79, 0.92],
                   [0.71, 0.92],
                   [0.78, 0.83],
                   [0.87, 0.92]])
class5 = np.array([[0.55, 0.4],
                   [0.66, 0.32],
                   [0.74, 0.49],
                   [0.89, 0.3],
                   [0.77, 0.02]])
class6 = np.array([[0.05, 0.15],
                   [0.09, 0.39],
                   [0.13, 0.51],
                   [0.25, 0.34],
                   [0.15, 0.36]])

def draw_classes():
    p1, = plt.plot(const.class1[:, 0], const.class1[:, 1], 'or')
    p2, = plt.plot(const.class2[:, 0], const.class2[:, 1], 'ob')
    p5, = plt.plot(const.class5[:, 0], const.class5[:, 1], 'ok')
    p6, = plt.plot(const.class6[:, 0], const.class6[:, 1], 'og')

    plt.legend([p1, p2, p5, p6], ["class1", "class2", "class3", "class4"])
    plt.xlabel("x")
    plt.ylabel("y")
    axes = plt.gca()
    plt.grid(True)
    axes.set_xlim([-0.1, 1.1])
    axes.set_ylim([-0.1, 1.1])


def draw_point(point, class_i):
    if class_i == 0:
        color = 'red'
    if class_i == 1:
        color = 'blue'
    if class_i == 2:
        color = "black"
    if class_i == 3:
        color = "green"

    plt.plot(point[0], point[1], marker='*', color=color)


def rbnf(classes, point):
    class_activities = np.array([])
    for i, class_i in enumerate(classes):
        class_activity = np.array([])
        for el in class_i:
            o = math.exp(-((el[0] - point[0]) ** 2 + (el[1] - point[1]) ** 2) / 0.1 ** 2)
            class_activity = np.append(class_activity, o)

        class_activities = np.append(class_activities, class_activity)
        class_activities = np.reshape(class_activities, (i + 1, -1))

    total_class_activities = np.array([])
    for activity in class_activities:
        total_class_activities = np.append(total_class_activities, sum(activity))

    max_activity_index = np.where(total_class_activities == np.amax(total_class_activities))[0]
    # print("class activities:\n{0}".format(class_activities))
    # print("total class activities:\n{0}".format(total_class_activities))
    # print("x: {0}, y: {1}, class: {2}".format(point[0], point[1], max_activity_index))

    return max_activity_index


def main():
    point = [0.42, 0.45]
    points = [[round(x, 2),round(y, 2)] for x in np.arange(0, 1.01, 0.01) for y in np.arange(0, 1.01, 0.01)]
    draw_classes()
    for p in points:
        point_class = rbnf([const.class1, const.class2, const.class5, const.class6], p)
        draw_point(p, point_class)

    point_class = rbnf([const.class1, const.class2, const.class5, const.class6], point)
    draw_point(point, point_class)

    plt.show()



if __name__ == '__main__':
    main()
