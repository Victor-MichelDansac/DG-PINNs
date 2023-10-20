"""
@author: Victor Michel-Dansac <victor.michel-dansac@inria.fr>
"""

import numpy as np


def u_ini(x, mesh):
    if mesh.source:
        a, b, c, u0 = mesh.a, mesh.b, mesh.c, mesh.u0
        return a * u0 / ((a + b * u0) * np.exp(-a * x / c) - b * u0)
    else:
        return 0.1 + np.exp(-((x - 0.5) ** 2) * 100) / 10


def dxu_ini(x, mesh):
    if mesh.source:
        a, b, c = mesh.a, mesh.b, mesh.c
        return (a * u_ini(x, mesh) + b * u_ini(x, mesh) ** 2) / c
    else:
        return -20 * (x - 0.5) * np.exp(-((x - 0.5) ** 2) * 100)


def u_exact(x, t, mesh):
    if mesh.source:
        return u_ini(x, mesh)
    else:
        xt = x - mesh.c * t
        xt = xt - np.floor(xt)
        return u_ini(xt, mesh)


def dxu_exact(x, t, mesh):
    if mesh.source:
        return dxu_ini(x, mesh)
    else:
        xt = x - mesh.c * t
        xt = xt - np.floor(xt)
        return dxu_ini(xt, mesh)
