"""
@author: Victor Michel-Dansac <victor.michel-dansac@inria.fr>
"""

import torch


def u_ini(x, mesh):
    if mesh.source:
        a, b, c, u0 = mesh.a, mesh.b, mesh.c, mesh.u0
        return a * u0 / ((a + b * u0) * torch.exp(-a * x / c) - b * u0)
    else:
        return 0.1 + torch.exp(-((x - 0.5) ** 2) * 100) / 10


def dxu_ini(x, mesh):
    if mesh.source:
        a, b, c = mesh.a, mesh.b, mesh.c
        return (a * u_ini(x, mesh) + b * u_ini(x, mesh) ** 2) / c
    else:
        return -20 * (x - 0.5) * torch.exp(-((x - 0.5) ** 2) * 100)


def u_exact(x, t, mesh):
    if mesh.source:
        try:  # case where t is a tensor and x is a float
            return u_ini(x, mesh) * torch.ones_like(t)
        except TypeError:  # case where t is a float and x is a tensor
            return u_ini(x, mesh)
    else:
        xt = x - mesh.c * t
        xt = xt - torch.floor(xt)
        return u_ini(xt, mesh)


def dxu_exact(x, t, mesh):
    if mesh.source:
        try:  # case where t is a tensor and x is a float
            return dxu_ini(x, mesh) * torch.ones_like(t)
        except TypeError:  # case where t is a float and x is a tensor
            return dxu_ini(x, mesh)
    else:
        xt = x - mesh.c * t
        xt = xt - torch.floor(xt)
        return dxu_ini(xt, mesh)
