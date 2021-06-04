# encoding: utf-8
"""
@author: Xin Zhang
@contact: zhangxin@szbl.ac.cn
@file: demo.py
@time: 6/2/21 10:49 AM
@desc: Circle Dynamics 圆圈动力学
    Relax the system of circles and container in Newton-mechanics to achieve a higher entropy state.
    Assume mol=1, time=1 in each iteration, then acceleration = force, V = V0 + a*t = V0 + force, X = X0 + V

    input: [W, L] wight and length of rectangle container
    input: [1, 2, 3, ...] radius of circles
    requirements: numpy, random, matplotlib
"""

import numpy as np
from random import uniform
from matplotlib import animation
import matplotlib.pyplot as plt


class Circle:
    def __init__(self, x, y, radius):
        self.radius = radius
        self.acceleration = np.array([0, 0])
        self.velocity = np.array([uniform(0, 1),  # v0
                                  uniform(0, 1)])
        self.position = np.array([x, y])

    @property
    def x(self):
        return self.position[0]

    @property
    def y(self):
        return self.position[1]

    @property
    def r(self):
        return self.radius

    def apply_force(self, force):
        # assume m=1, then acceleration=force
        self.acceleration = np.add(self.acceleration, force)

    def update(self):
        # assume t=1, then v=v0+a*1
        self.velocity = np.add(self.velocity, self.acceleration)
        # x=x0+v*1
        self.position = np.add(self.position, self.velocity)
        self.acceleration = 0


class Pack:
    def __init__(self, width, height, list_circles):

        self.iter = 0
        self.list_circles = list_circles  # list(Circle)
        self.right_border = width
        self.upper_border = height

        # [[x, y], [x, y], ....] denote the force of collision with neighbors, added by each separate force.
        self.list_separate_forces = [np.array([0, 0])] * len(self.list_circles)
        self.list_near_circle = [0] * len(self.list_circles)  # denote num of neighbors of each circle

    @property
    def get_list_circle(self):
        return self.list_circles

    def _normalize(self, v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    def run(self):
        # run one times, called externally like by matplotlib.animation
        self.iter += 1
        for circle in self.list_circles:
            self.check_borders(circle)
            self.apply_separation_forces_to_circle(circle)
            # self.check_circle_positions(circle)

    def check_borders(self, circle):
        orientation = np.array([0, 0])  # orientation of reaction force which perpendicular to collision surface
        if circle.x <= 0 + circle.r:
            orientation = np.add([1, 0], orientation)

        if circle.x >= self.right_border - circle.r:
            orientation = np.add([-1, 0], orientation)

        if circle.y <= 0 + circle.r:
            orientation = np.add([0, 1], orientation)

        if circle.y >= self.upper_border - circle.r:
            orientation = np.add([0, -1], orientation)

        react_orientation = self._normalize(orientation)
        # magnitude of reaction v: projection * react_orientation
        react_separate_v = np.dot(circle.velocity, react_orientation) * react_orientation

        w = np.subtract(circle.velocity, react_separate_v)
        circle.velocity = np.subtract(w, react_separate_v)
        circle.update()

    def check_circle_positions(self, circle):
        # determined the circle when to stop
        i = self.list_circles.index(circle)

        # for neighbour in list_neighbours:
        # If we execute a full loops, we'd compare every circle to every other particle twice over (and compare each
        # particle to itself), because i had compared with i-1 by its i-1 neighbor.
        for neighbour in self.list_circles[i + 1:]:
            d = self._distance_circles(circle, neighbour)
            if d < (circle.r + neighbour.r):
                # keep it moving
                return
        # if no-touching with others then let it rest
        circle.velocity[0] = 0
        circle.velocity[1] = 0

    def _get_separation_force(self, c1, c2):
        steer = np.array([0, 0])
        d = self._distance_circles(c1, c2)
        if 0 < d < (c1.r + c2.r):
            # orientate to c1, means the force of c1, and the force of c2 is opposite
            diff = np.subtract(c1.position, c2.position)
            diff = self._normalize(diff)  # orientation
            diff = np.divide(diff, 1 / d ** 2)  # magnitude of force is related to distance
            # r = min(c1.r, c2.r)
            # diff = (r/d)**3 * diff
            steer = np.add(steer, diff)
        return steer

    def _distance_circles(self, c1, c2):
        x1, y1 = c1.x, c1.y
        x2, y2 = c2.x, c2.y
        dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return dist

    def apply_separation_forces_to_circle(self, circle):
        i = self.list_circles.index(circle)

        for neighbour in self.list_circles[i + 1:]:
            j = self.list_circles.index(neighbour)
            force_ij = self._get_separation_force(circle, neighbour)

            if np.linalg.norm(force_ij) > 0:
                self.list_separate_forces[i] = np.add(self.list_separate_forces[i], force_ij)  # vector addition
                self.list_separate_forces[j] = np.subtract(self.list_separate_forces[j], force_ij)  # vector addition
                self.list_near_circle[i] += 1
                self.list_near_circle[j] += 1

        # resultant force from neighbors of this circle
        # if np.linalg.norm(self.list_separate_forces[i]) > 0:
        #     self.list_separate_forces[i] = np.subtract(self.list_separate_forces[i], circle.velocity)

        if self.list_near_circle[i] > 0:
            self.list_separate_forces[i] = np.divide(self.list_separate_forces[i], 20*self.list_near_circle[i])

        separation = self.list_separate_forces[i]
        circle.apply_force(separation)
        circle.update()


def animate(i):
    patches = []
    p.run()
    # fig.clf()
    for c in list_circles:
        circle = plt.Circle((c.x, c.y), radius=c.r, fill=False)
        patches.append(plt.gca().add_patch(circle))
    return patches


if __name__ == '__main__':
    list_circles = list()
    for i in range(2):
        # generate new circles
        list_circles.append(Circle(40, 40, 5))
    p = Pack(width=80, height=80, list_circles=list_circles)

    fig = plt.figure()
    plt.axis('scaled')
    plt.xlim(-20, 100)
    plt.ylim(-20, 100)
    rectangle = plt.Rectangle((0, 0), width=80, height=80, fc='none', ec='k')
    plt.gca().add_patch(rectangle)

    anim = animation.FuncAnimation(fig, animate,
                                   frames=500, interval=2, blit=True)
    plt.show()
    anim.save('animation.gif', dpi=80, writer='imagemagick')

