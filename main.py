from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from numpy import sqrt, float64

# import pygame
# from pygame.locals import *
from OpenGL.GL import *
# from OpenGL.GLU import *


ELEMENT_INDEX = int

edges = (
    (0, 1),
    (0, 3),
    (0, 4),
    (2, 1),
    (2, 3),
    (2, 7),
    (6, 3),
    (6, 4),
    (6, 7),
    (5, 1),
    (5, 4),
    (5, 7)
)


def draw_cube(verticies):
    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glVertex3fv(verticies[vertex])
    glEnd()


class Point:
    def __init__(self, x, y, z):
        self.x = float64(x)
        self.y = float64(y)
        self.z = float64(z)

    def __repr__(self):
        return f"({self.x}, {self.y}, {self.z})"

    def __hash__(self):
        return hash((self.x, self.y, self.z))

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z

    @staticmethod
    def get_point_between(point_a, point_b):
        x = (point_a.x + point_b.x) / 2
        y = (point_a.y + point_b.y) / 2
        z = (point_a.z + point_b.z) / 2
        return Point(x, y, z)


class Parallelepiped:
    def __init__(self, ax, ay, az, nx: int, ny: int, nz: int):
        self.ax = float64(ax)
        self.ay = float64(ay)
        self.az = float64(az)

        self.nx = nx
        self.ny = ny
        self.nz = nz

        self.delta_x = self.ax / self.nx
        self.delta_y = self.ay / self.ny
        self.delta_z = self.az / self.nz

        self.count_of_elements = nx * ny * nz

        self.akt = self.build_akt()
        self.elements, self.nt = self.build_elements_nt()

        len_akt = 4 * nx * ny * nz + 3 * (nx * ny + ny * nz + nx * nz) + 2 * (nx + ny + nz) + 1
        if len_akt != len(self.akt):
            raise Exception("Wrong count of vertexes")

        self.DFIABG = self.build_dfiabg()

        self.DJs = self.build_delta()

        print("Init completed")

    def build_elements_nt(self) -> Tuple[Dict[ELEMENT_INDEX, List[Point]], Dict[ELEMENT_INDEX, List[int]]]:
        elements = defaultdict(list)
        nt = defaultdict(list)
        for k in range(self.nz):
            for j in range(self.ny):
                for i in range(self.nx):
                    el_index = i + self.nx * j + self.nx * self.ny * k

                    node1 = Point(x=i * self.delta_x, y=j * self.delta_y, z=k * self.delta_z)
                    node2 = Point(node1.x + self.delta_x, node1.y, node1.z)
                    node3 = Point(node1.x + self.delta_x, node1.y + self.delta_y, node1.z)
                    node4 = Point(node1.x, node1.y + self.delta_y, node1.z)
                    nt[el_index].append(self.find_akt_index(node1))
                    nt[el_index].append(self.find_akt_index(node2))
                    nt[el_index].append(self.find_akt_index(node3))
                    nt[el_index].append(self.find_akt_index(node4))

                    node5 = Point(node1.x, node1.y, node1.z + self.delta_z)
                    node6 = Point(node1.x + self.delta_x, node1.y, node1.z + self.delta_z)
                    node7 = Point(node1.x + self.delta_x, node1.y + self.delta_y, node1.z + self.delta_z)
                    node8 = Point(node1.x, node1.y + self.delta_y, node1.z + self.delta_z)
                    nt[el_index].append(self.find_akt_index(node5))
                    nt[el_index].append(self.find_akt_index(node6))
                    nt[el_index].append(self.find_akt_index(node7))
                    nt[el_index].append(self.find_akt_index(node8))

                    node9 = Point.get_point_between(node1, node2)
                    node10 = Point.get_point_between(node2, node3)
                    node11 = Point.get_point_between(node3, node4)
                    node12 = Point.get_point_between(node4, node1)
                    nt[el_index].append(self.find_akt_index(node9))
                    nt[el_index].append(self.find_akt_index(node10))
                    nt[el_index].append(self.find_akt_index(node11))
                    nt[el_index].append(self.find_akt_index(node12))

                    node13 = Point.get_point_between(node1, node5)
                    node14 = Point.get_point_between(node2, node6)
                    node15 = Point.get_point_between(node3, node7)
                    node16 = Point.get_point_between(node4, node8)
                    nt[el_index].append(self.find_akt_index(node13))
                    nt[el_index].append(self.find_akt_index(node14))
                    nt[el_index].append(self.find_akt_index(node15))
                    nt[el_index].append(self.find_akt_index(node16))

                    node17 = Point.get_point_between(node5, node6)
                    node18 = Point.get_point_between(node6, node7)
                    node19 = Point.get_point_between(node7, node8)
                    node20 = Point.get_point_between(node8, node5)
                    nt[el_index].append(self.find_akt_index(node17))
                    nt[el_index].append(self.find_akt_index(node18))
                    nt[el_index].append(self.find_akt_index(node19))
                    nt[el_index].append(self.find_akt_index(node20))

                    elements[el_index] = [
                        node1, node2, node3, node4, node5, node6, node7, node8, node9, node10, node11, node12, node13,
                        node14, node15, node16, node17, node18, node19, node20
                    ]

        return elements, nt

    def build_akt(self):
        akt = list()
        for k in range(2 * self.nz + 1):
            if (k % 2) == 0:
                for j in range(2 * self.ny + 1):
                    if (j % 2) == 0:
                        for i in range(2 * self.nx + 1):
                            akt.append(Point(x=i * self.delta_x / 2, y=j * self.delta_y / 2, z=k * self.delta_z / 2))
                    else:
                        # if (j % 2) == 1:
                        for i in range(self.nx + 1):
                            akt.append(Point(x=i * self.delta_x, y=j * self.delta_y / 2, z=k * self.delta_z / 2))
            else:
                # (k % 2) == 1:
                for j in range(self.ny + 1):
                    for i in range(self.nx + 1):
                        akt.append(Point(x=i * self.delta_x, y=j * self.delta_y, z=k * self.delta_z / 2))
        return akt

    def find_akt_index(self, node_to_find: Point):
        for i, node in enumerate(self.akt):
            if node == node_to_find:
                return i
        raise Exception("Could not find the Point in AKT")

    def build_dfiabg(self):
        DFIABG = np.zeros((27, 3, 20))
        AiBiGi = [  # вузли в локальній системі координат (A, B, G)
            [-1, 1, -1], [1, 1, -1], [1, -1, -1], [-1, -1, -1],
            [-1, 1, 1], [1, 1, 1], [1, -1, 1], [-1, -1, 1],
            [0, 1, -1], [1, 0, -1], [0, -1, -1], [-1, 0, -1],
            [-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0],
            [0, 1, 1], [1, 0, 1], [0, -1, 1], [-1, 0, 1],
        ]
        ABG_CONST = (-sqrt(0.6), 0, sqrt(0.6))  # точки Гауса
        node_index = 0
        for A in ABG_CONST:
            for B in ABG_CONST:
                for G in ABG_CONST:
                    # print(f"({A}, {B}, {G})")
                    for i in range(0, 8):
                        # calculate d_fi for i = 0, 7
                        a_i, b_i, g_i = tuple(AiBiGi[i])
                        DFIABG[node_index, 0, i] = (  # d_fi_A
                            0.125 * (1 + B * b_i) * (1 + G * g_i) * (a_i * (2 * A * a_i + B * b_i + G * g_i - 1))
                        )
                        DFIABG[node_index, 1, i] = (  # d_fi_B
                            0.125 * (1 + A * a_i) * (1 + G * g_i) * (b_i * (A * a_i + 2 * B * b_i + G * g_i - 1))
                        )
                        DFIABG[node_index, 2, i] = (  # d_fi_G
                            0.125 * (1 + B * b_i) * (1 + A * a_i) * (g_i * (A * a_i + B * b_i + 2 * G * g_i - 1))
                        )

                    for i in range(8, 20):
                        # calculate d_fi for i = 8, 19
                        a_i, b_i, g_i = tuple(AiBiGi[i])

                        DFIABG[node_index, 0, i] = (  # d_fi_A
                            0.25 * (1 + B * b_i) * (1 + G * g_i) * (-a_i ** 3 * b_i ** 2 * G ** 2 - a_i ** 3 * B ** 2 * g_i ** 2 - 3 * A ** 2 * a_i * b_i ** 2 * g_i ** 2 + a_i - 2 * A * b_i ** 2 * g_i ** 2)
                        )
                        DFIABG[node_index, 1, i] = (  # d_fi_B
                            0.25 * (1 + A * a_i) * (1 + G * g_i) * (-a_i ** 2 * b_i ** 3 * G ** 2 - A ** 2 * b_i ** 3 * g_i ** 2 - 3 * a_i ** 2 * B ** 2 * b_i * g_i ** 2 + b_i - 2 * a_i ** 2 * B * g_i ** 2)
                        )
                        DFIABG[node_index, 2, i] = (  # d_fi_G
                            0.25 * (1 + B * b_i) * (1 + A * a_i) * (-a_i ** 2 * B ** 2 * g_i ** 3 - A ** 2 * b_i ** 2 * g_i ** 3 - 3 * a_i ** 2 * b_i ** 2 * G ** 2 * g_i + g_i - 2 * a_i ** 2 * b_i ** 2 * G)
                        )
                    node_index += 1
        return DFIABG

    def build_delta(self):  # побудова якубіанів
        DJs = defaultdict(list)
        for n, element in enumerate(self.elements):
            for k in range(27):
                DJ = np.zeros((3, 3))
                for abg in range(3):
                    for i in range(20):
                        global_node_index = self.nt.get(n)[i]
                        node = self.akt[global_node_index]
                        dfiabg = self.DFIABG[k, abg, i]
                        DJ[abg, 0] += node.x * dfiabg  # Dx/Da, Dx/Db, Dx/Dg
                        DJ[abg, 1] += node.y * dfiabg  # Dy ...
                        DJ[abg, 2] += node.z * dfiabg  # Dz ...

                DJs[n].append(DJ)
        return DJs


def main():
    # ax = int(input("ax="))
    # ay = int(input("ay="))
    # az = int(input("az="))

    ax, ay, az = 2, 2, 2

    global_verticies = (
        (ax, 0, 0), (ax, ay, 0), (0, ay, 0), (0, 0, 0), (ax, 0, az), (ax, ay, az), (0, 0, az), (0, ay, az),
    )

    # nx = int(input("nx="))
    # ny = int(input("ny="))
    # nz = int(input("nz="))

    nx, ny, nz = 2, 1, 2

    obj = Parallelepiped(ax, ay, az, nx, ny, nz)

    # verticies_by_element = list()
    # for iz in range(nz + 1):
    #     for iy in range(ny + 1):
    #         for ix in range(nx + 1):
    #             verticies = (
    #                 (len_nx * ix, 0, 0), (len_nx * ix, len_ny * iy, 0),
    #                 (0, len_ny * iy, 0), (0, 0, 0),
    #                 (len_nx * ix, 0, len_nz * iz), (len_nx * ix, len_ny * iy, len_nz * iz),
    #                 (0, 0, len_nz * iz), (0, len_ny * iy, len_nz * iz),
    #             )
    #             verticies_by_element.append(verticies)
    #
    # pygame.init()
    # display_size = (800, 600)
    # pygame.display.set_mode(display_size, DOUBLEBUF | OPENGL)
    # gluPerspective(45, (display_size[0]/display_size[1]), 0.1, 100.0)
    # glTranslatef(0, -5, -45)
    #
    # while True:
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             pygame.quit()
    #             quit()
    #
    #     glRotatef(1, 0, 1, 0)
    #     glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    #     # draw_cube(global_verticies)
    #     for vert in verticies_by_element:
    #         draw_cube(vert)
    #     pygame.display.flip()
    #     pygame.time.wait(10)


if __name__ == "__main__":
    main()
