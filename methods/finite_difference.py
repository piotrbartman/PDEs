"""
Created at 19.01.2020

@author: Piotr Bartman
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import scipy as sc
import scipy.sparse
import scipy.sparse.linalg


class Grid1d:
    def __init__(self, setup):
        self.x_num = round((setup.x_range[1] - setup.x_range[0]) / setup.dx)
        self.u = np.zeros((self.x_num + 1, 1))
        self.x, setup.dx = np.linspace(setup.x_range[0], setup.x_range[1], self.x_num + 1, retstep=True)


class Setup1:
    x_range = (0.0, 1.0)
    dx = 0.01
    y_range = (0.0, 1.0)
    dy = 0.01
    boundary = 0.0

    @property
    def r(self):
        return self.dt / self.dx ** 2

    @staticmethod
    def f(x, y):
        return -2 * np.pi ** 2 * np.sin(np.pi * x) * np.sin(np.pi * y)

    @staticmethod
    def exact(x, y):
        return np.sin(np.pi * x) * np.sin(np.pi * y)


class Setup2:
    x_range = (0.0, 1.0)
    dx = 0.01
    boundary = (lambda t: 0.0, lambda t: 0.0)
    t_range = (0.0, 2.0)
    dt = 0.00001
    alpha = 0.0

    @property
    def r(self):
        return self.dt / self.dx ** 2

    @staticmethod
    def initial(x):
        return np.sin(np.pi * x)

    @staticmethod
    def exact(t, x):
        return np.exp(-np.pi ** 2 * t) * np.sin(np.pi * x)


class Setup3:
    x_range = (0.0, 1.0)
    dx = 0.01
    boundary = (lambda t: 0.0, lambda t: 0.0)
    t_range = (0.0, 2.0)
    dt = 0.00001
    alpha = 0.0

    @property
    def r(self):
        return self.dt ** 2 / self.dx ** 2

    @staticmethod
    def initial(x):
        return 0.125 * np.sin(np.pi * x)

    @staticmethod
    def initial_der(x):
        return 0

    @staticmethod
    def exact(t, x):
        return 0.125 * np.sin(np.pi * x) * np.cos(np.pi * t)


def create_matrices(setup):
    grid = Grid1d(setup)
    u, x, dx = grid.u, grid.x, setup.dx
    for i, xi in enumerate(x): u[i] = setup.initial(xi)
    u[0] = setup.boundary[0](0)
    u[-1] = setup.boundary[-1](0)
    t_num = round((setup.t_range[1] - setup.t_range[0]) / setup.dt)
    t, setup.dt = np.linspace(setup.t_range[0], setup.t_range[1], t_num, retstep=True)
    r = setup.r
    d = np.zeros((grid.x_num - 1, 1), 'd')

    diagonals = np.zeros((3, grid.x_num - 1))
    diagonals[0, :] = -r * setup.alpha
    diagonals[1, :] = 1 + 2 * r * setup.alpha
    diagonals[2, :] = -r * setup.alpha
    As = sc.sparse.spdiags(diagonals, [-1, 0, 1], grid.x_num - 1, grid.x_num - 1, format='csc')

    return As, grid, t


def create_RH(u, setup, grid):
    d = np.zeros((grid.x_num - 1, 1), 'd')
    r = setup.dt / setup.dx ** 2
    d[:] = u[1:-1] + r * (1 - setup.alpha) * (u[0:-2] - 2 * u[1:-1] + u[2:])
    d[0] += r * u[0] * setup.alpha
    d[-1] += r * u[-1] * setup.alpha
    return d


def create_RH3(u, setup, grid, u_old):
    d = np.zeros((grid.x_num - 1, 1), 'd')
    r = setup.r
    d[:] = 2 * u[1:-1] + r * (1 - 2 * setup.alpha) * (u[0:-2] - 2 * u[1:-1] + u[2:])
    d[:] += -u_old[1:-1] + r * setup.alpha * (u_old[0:-2] - 2 * u_old[1:-1] + u_old[2:])
    d[0] += r * u[0] * setup.alpha
    d[-1] += r * u[-1] * setup.alpha
    return d


def solve(A, u, t, setup, grid):
    errors = []
    for _t in t:
        d = create_RH(u, setup, grid)
        w = sc.sparse.linalg.spsolve(A, d)
        e = np.zeros_like(u)
        for i, xi in enumerate(grid.x):
            e[i] = setup.exact(_t, xi)
        errors.append(np.mean((u - e) ** 2))
        u[1:-1] = w[:, None]
    return errors


def fictional(x, setup, hx, ht):
    return setup.initial(x) - ht * setup.initial_der(x) + ht ** 2 / 2 * \
           (setup.initial(x - hx) - 2 * setup.initial(x) + setup.initial(x + hx)) / hx ** 2


def solve3(A, u, u_old, t, setup, grid):
    errors = []
    for _t in t:
        d = create_RH3(u, setup, grid, u_old)
        w = sc.sparse.linalg.spsolve(A, d)
        e = np.zeros_like(u)
        for i, xi in enumerate(grid.x):
            e[i] = setup.exact(_t, xi)
        errors.append(np.mean((u - e) ** 2))
        u_old[:] = u
        u[1:-1] = w[:, None]
    return errors


def scheme1(dx, dy):
    setup = Setup1()
    setup.dx, setup.dy = dx, dy
    #     A, grid, t = create_matrices(setup)
    x_num = int(setup.x_range[-1] / setup.dx)
    y_num = int(setup.y_range[-1] / setup.dy)
    setup.dx = setup.x_range[-1] / x_num
    setup.dy = setup.y_range[-1] / y_num

    from scipy.sparse import lil_matrix
    size = (x_num + 1) * (y_num + 1)
    A = lil_matrix((size, size))
    hx = 1 / dx ** 2
    hy = 1 / dy ** 2
    r = -2 * (hx + hy)

    d = np.zeros((size, 1), 'd')
    s = y_num + 1
    for i in range(0, x_num + 1):
        for j in range(0, y_num + 1):
            k = i * s + j
            if i == 0 or i == x_num or j == 0 or j == y_num:
                A[k, k] = 1
                d[k] = 0
            else:
                A[k, k + s] = hx
                A[k, k + 1] = hy
                A[k, k - s] = hx
                A[k, k - 1] = hy
                A[k, k] = r
                d[k] = setup.f(i * dx, j * dy)
    w = sc.sparse.linalg.spsolve(A, d)
    _d = A.dot(w)
    # print("prec", np.linalg.norm(d.T - _d, 1))
    e = np.zeros_like(w)
    #     print("solution\n", w)
    for i in range(0, x_num + 1):
        for j in range(0, y_num + 1):
            e[i * s + j] = setup.exact(i * dx, j * dy)
    #     print("exact\n", e)
    # print("prec", np.linalg.norm(d.T - A.dot(e), 1))
    return  np.mean((w - e) ** 2)


def scheme2(alpha, dx, dt):
    setup = Setup2()
    setup.alpha, setup.dx, setup.dt = alpha, dx, dt
    A, grid, t = create_matrices(setup)
    errors = solve(A, grid.u, t, setup, grid)
    return np.mean(errors)


def scheme3(alpha, dx, dt):
    setup = Setup3()
    setup.alpha, setup.dx, setup.dt = alpha, dx, dt
    A, grid, t = create_matrices(setup)
    u_old = np.zeros_like(grid.u)
    for i, xi in enumerate(grid.x):
        u_old[i] = fictional(xi, setup, dx, dt)
    errors = solve3(A, grid.u, u_old, t, setup, grid)
    return np.mean(errors)