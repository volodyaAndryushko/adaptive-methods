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
        print("Built akt")
        self.elements, self.nt = self.build_elements_nt()
        self.ZP = self.build_zp()  # тиснемо на верхні грані всіх верхніх елементів

        len_akt = 4 * nx * ny * nz + 3 * (nx * ny + ny * nz + nx * nz) + 2 * (nx + ny + nz) + 1
        if len_akt != len(self.akt):
            raise Exception("Wrong count of vertexes")
        print("Built elements & nt")

        self.DFIABG = self.build_dfiabg()
        print("Built DFIABG")

        self.DJs = self.build_delta()
        print("Built DJs")

        self.DFIXYZ = self.build_dfixyz()
        print("Built DFIXYZ")

        self.MGE = self.build_mge()
        print("Built MGE")

        self.DPSITE = self.build_dpsite()
        print("Built DPSITE")

        self.DXYZTE = self.build_dxyzte()
        print("Built DXYZTE")

        self.PSI = self.build_psi()
        print("Built PSI")

        self.FE = self.build_fe()
        print("Built FE")

        self.K = self.build_K()
        print("Built K")

        self.F = self.build_F()
        print("Built F")

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

    def build_zp(self):
        count_elements_under_pressure = self.nx * self.ny
        count_free_elements = self.count_of_elements - count_elements_under_pressure
        ZP = dict()
        for element_index in range(self.count_of_elements - 1, count_free_elements - 1, -1):
            ZP[element_index] = 5  # todo: supports only one edge to be pressed
        return ZP

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

    def build_dfixyz(self):
        DFIXYZ_by_element = defaultdict(list)
        for n, element in enumerate(self.elements):
            for k in range(27):
                DFIXYZ = list()
                DJ = self.DJs[n][k]
                for i in range(20):
                    # dfiabg = [DfiDa, DfiDb, DfiDg]
                    dfiabg = np.array([self.DFIABG[k, 0, i], self.DFIABG[k, 1, i], self.DFIABG[k, 2, i]])

                    # Розв'язуємо систему лінійних рівнянь
                    dfixyz = np.linalg.solve(DJ, dfiabg)  # DfiDx, DfiDy, DfiDz
                    DFIXYZ.append(dfixyz)
                DFIXYZ_by_element[n].append(DFIXYZ)
        return DFIXYZ_by_element

    def build_mge(self):
        mge_by_element = dict()
        C_CONST = (float64(5 / 9), float64(8 / 9), float64(5 / 9))
        E = float64(70)  # Значення модуля Юнга для Алюмінію
        v = float64(0.34)  # Значення коефіцієнта Пуассона для Алюмінію
        _lambda = E / ((1 + v) * (1 - 2 * v))  # вікіпедія каже (E * v) / ...
        m = E / (2 * (1 + v))

        for n, element in enumerate(self.elements):
            a_11, a_22, a_33, a_12, a_13, a_23 = (np.zeros((20, 20)) for i in range(6))
            for i in range(20):
                for j in range(20):
                    counter = 0
                    for c_m in C_CONST:
                        for c_n in C_CONST:
                            for c_k in C_CONST:
                                delta = np.linalg.det(self.DJs[n][counter])
                                common_value = c_m * c_n * c_k * delta
                                dfi_i_dx = self.DFIXYZ[n][counter][i][0]
                                dfi_j_dx = self.DFIXYZ[n][counter][j][0]
                                dfi_i_dy = self.DFIXYZ[n][counter][i][1]
                                dfi_j_dy = self.DFIXYZ[n][counter][j][1]
                                dfi_i_dz = self.DFIXYZ[n][counter][i][2]
                                dfi_j_dz = self.DFIXYZ[n][counter][j][2]

                                a_11[i, j] += common_value * (_lambda * (1 - v) * dfi_i_dx * dfi_j_dx + m * (dfi_i_dy * dfi_j_dy + dfi_i_dz * dfi_j_dz))
                                a_22[i, j] += common_value * (_lambda * (1 - v) * dfi_i_dy * dfi_j_dy + m * (dfi_i_dx * dfi_j_dx + dfi_i_dz * dfi_j_dz))
                                a_33[i, j] += common_value * (_lambda * (1 - v) * dfi_i_dz * dfi_j_dz + m * (dfi_i_dx * dfi_j_dx + dfi_i_dy * dfi_j_dy))
                                a_12[i, j] += common_value * _lambda * v * dfi_i_dx * dfi_j_dy + m * dfi_i_dy * dfi_j_dx
                                a_13[i, j] += common_value * _lambda * v * dfi_i_dx * dfi_j_dz + m * dfi_i_dz * dfi_j_dx
                                a_23[i, j] += common_value * _lambda * v * dfi_i_dy * dfi_j_dz + m * dfi_i_dz * dfi_j_dy
                                counter += 1

            mge = np.zeros((60, 60))
            mge[:20, :20] = a_11
            mge[20:40, 20:40] = a_22
            mge[40:, 40:] = a_33
            mge[:20, 20:40] = a_12
            mge[20:40, :20] = a_12.T

            mge[:20, 40:] = a_13
            mge[40:, :20] = a_13.T

            mge[20:40, 40:] = a_23
            mge[40:, 20:40] = a_23.T
            mge_by_element[n] = mge
            # assert np.allclose(mge, mge.T, rtol=1e-05, atol=1e-08)  # check symmetric
        return mge_by_element

    def build_dpsite(self):
        DPSITE = np.zeros((9, 2, 8))
        EiTi = [  # вузли в локальній системі координат (E, T)
            [-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0],
        ]
        ABG_CONST = (-sqrt(0.6), 0, sqrt(0.6))  # точки Гауса
        node_index = 0
        for E in ABG_CONST:
            for T in ABG_CONST:
                # print(f"({A}, {B}, {G})")
                for i in range(0, 8):
                    e_i, t_i = tuple(EiTi[i])
                    if i < 4:
                        d_psi_E = (
                            0.25 * (T * t_i + 1) * (e_i * (e_i * E + T * t_i - 1) + e_i * (e_i * E + 1))
                        )
                        d_psi_T = (
                            0.25 * (E * e_i + 1) * (t_i * (t_i * T + E * e_i - 1) + t_i * (t_i * T + 1))
                        )
                    elif i in (4, 6):
                        d_psi_E = (-T * t_i - 1) * E
                        d_psi_T = 0.5 * (1 - E ** 2) * t_i
                    else:   # if i in (5, 7):
                        d_psi_E = 0.5 * (1 - T ** 2) * e_i
                        d_psi_T = (-E * e_i - 1) * T

                    DPSITE[node_index, 0, i] = d_psi_E
                    DPSITE[node_index, 1, i] = d_psi_T
                node_index += 1
        return DPSITE

    def build_dxyzte(self):
        FACE_INDEX_POINTS = {
            5: {4, 5, 6, 7, 16, 17, 18, 19},
        }
        DXYZTE = defaultdict(list)
        for n, face_index in self.ZP.items():
            for counter in range(9):
                dxyzte = np.zeros((3, 2))
                for i, local_node_index in enumerate(FACE_INDEX_POINTS[face_index]):  # for i in range(8)
                    global_node_index = self.nt[n][local_node_index]
                    global_node = self.akt[global_node_index]
                    dpsi_de = self.DPSITE[counter][0][i]
                    dpsi_dt = self.DPSITE[counter][1][i]

                    dxyzte[0][0] += global_node.x * dpsi_de  # dx/de
                    dxyzte[1][0] += global_node.y * dpsi_de  # dy/de
                    dxyzte[2][0] += global_node.z * dpsi_de  # dz/de

                    dxyzte[0][1] += global_node.x * dpsi_dt  # dx/dt
                    dxyzte[1][1] += global_node.y * dpsi_dt  # dy/dt
                    dxyzte[2][1] += global_node.z * dpsi_dt  # dz/dt

                DXYZTE[n].append(dxyzte)
        return DXYZTE

    def build_psi(self):
        ABG_CONST = (-sqrt(0.6), 0, sqrt(0.6))  # точки Гауса
        EiTi = [  # вузли в локальній системі координат (E, T)
            [-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0],
        ]
        PSI = dict()
        for n, face_index in self.ZP.items():
            psi = np.zeros((9, 8))
            counter = 0
            for E in ABG_CONST:
                for T in ABG_CONST:
                    for i in range(8):
                        e_i, t_i = EiTi[i]
                        if i < 4:
                            psi[counter][i] = 0.25 * (1 + E * e_i) * (1 + T * t_i) * (E * e_i + T * t_i - 1)
                        elif i in (4, 6):
                            psi[counter][i] = 0.5 * (1 - E ** 2) * (1 + T * t_i)
                        else:  # if i in (5, 7):
                            psi[counter][i] = 0.5 * (1 - T ** 2) * (1 + E * e_i)
                    counter += 1
            PSI[n] = psi
        return PSI

    def build_fe(self):
        Fe = dict()
        FACE_INDEX_POINTS = {
            5: [4, 5, 6, 7, 16, 17, 18, 19],
        }
        C_CONST = (float64(5 / 9), float64(8 / 9), float64(5 / 9))
        P = float64(0.2)  # Значення навантаження  # todo: move to input

        for n, element in enumerate(self.elements):
            f_1, f_2, f_3 = (np.zeros(20) for i in range(3))
            if n in self.ZP.keys():
                edge_index = self.ZP[n]
                for i in range(8):
                    counter = 0
                    for c_m in C_CONST:
                        for c_n in C_CONST:
                            dx_de = self.DXYZTE[n][counter][0][0]
                            dy_de = self.DXYZTE[n][counter][1][0]
                            dz_de = self.DXYZTE[n][counter][2][0]
                            dx_dt = self.DXYZTE[n][counter][0][1]
                            dy_dt = self.DXYZTE[n][counter][1][1]
                            dz_dt = self.DXYZTE[n][counter][2][1]
                            psi = self.PSI[n][counter][i]

                            local_node_index = FACE_INDEX_POINTS[edge_index][i]
                            common_value = c_m * c_n * P
                            f_1[local_node_index] += common_value * (dy_de * dz_dt - dz_de * dy_dt) * psi
                            f_2[local_node_index] += common_value * (dz_de * dx_dt - dx_de * dz_dt) * psi
                            f_3[local_node_index] += common_value * (dx_de * dy_dt - dy_de * dx_dt) * psi
                            counter += 1
                Fe[n] = np.zeros(60)
                Fe[n][:20] = f_1
                Fe[n][20:40] = f_2
                Fe[n][40:] = f_3
            else:
                Fe[n] = np.zeros(60)
        return Fe

    def build_K(self):
        count_of_nodes = len(self.akt)
        K = np.zeros((3 * count_of_nodes, 3 * count_of_nodes))
        for n, Ke in self.MGE.items():
            for i, _ in enumerate(Ke):
                for j, _ in enumerate(Ke[i]):
                    k = 3 * self.nt[n][i % 20] + i // 20
                    l = 3 * self.nt[n][j % 20] + j // 20
                    K[k][l] = Ke[i][j]
        return K  # some items == 0 ;(  (I hope that's OK)

    def build_F(self):
        count_of_nodes = len(self.akt)
        F = np.zeros(3 * count_of_nodes)
        for n, Fe in self.FE.items():
            for i, _ in enumerate(Fe):
                k = 3 * self.nt[n][i % 20] + i // 20
                F[k] = Fe[i]
        return F


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
