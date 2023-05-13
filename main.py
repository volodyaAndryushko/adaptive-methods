import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *


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


def main():
    # ax = int(input("ax="))
    # ay = int(input("ay="))
    # az = int(input("az="))

    ax, ay, az = 13, 13, 13

    global_verticies = (
        (ax, 0, 0), (ax, ay, 0), (0, ay, 0), (0, 0, 0), (ax, 0, az), (ax, ay, az), (0, 0, az), (0, ay, az),
    )

    # nx = int(input("nx="))
    # ny = int(input("ny="))
    # nz = int(input("nz="))

    nx, ny, nz = 4, 4, 4

    len_nx = ax / nx
    len_ny = ay / ny
    len_nz = az / nz

    verticies_by_element = list()
    for iz in range(nz + 1):
        for iy in range(ny + 1):
            for ix in range(nx + 1):
                verticies = (
                    (len_nx * ix, 0, 0), (len_nx * ix, len_ny * iy, 0),
                    (0, len_ny * iy, 0), (0, 0, 0),
                    (len_nx * ix, 0, len_nz * iz), (len_nx * ix, len_ny * iy, len_nz * iz),
                    (0, 0, len_nz * iz), (0, len_ny * iy, len_nz * iz),
                )
                verticies_by_element.append(verticies)

    pygame.init()
    display_size = (800, 600)
    pygame.display.set_mode(display_size, DOUBLEBUF | OPENGL)
    gluPerspective(45, (display_size[0]/display_size[1]), 0.1, 100.0)
    glTranslatef(-2, -2, -45)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        glRotatef(1, 0, 1, 0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        # draw_cube(global_verticies)
        for vert in verticies_by_element:
            draw_cube(vert)
        pygame.display.flip()
        pygame.time.wait(10)


if __name__ == "__main__":
    main()
