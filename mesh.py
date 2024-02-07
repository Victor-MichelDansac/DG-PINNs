"""
@author: Victor Michel-Dansac <victor.michel-dansac@inria.fr>
"""

import time

import DG_scheme
import matplotlib.pyplot as plt
import torch
from model import Network


class Mesh:
    time_integrators = {
        1: "Explicit Euler",
        2: "SSPRK2",
        3: "SSPRK35",
        4: "SSPRK410",
    }

    cfl_factors = {
        "Explicit Euler": 1,
        "SSPRK2": 1,
        "SSPRK3": 1,
        "SSPRK34": 2,
        "SSPRK35": 2.65,
        "SSPRK45": 1.508,
        "SSPRK410": 3,
    }
    DEFAULT_A_MIN, DEFAULT_A_MAX = 0.5, 1
    DEFAULT_B_MIN, DEFAULT_B_MAX = 0.5, 1
    DEFAULT_U0_MIN, DEFAULT_U0_MAX = 0.1, 0.2

    def __init__(
        self,
        category,
        nx,
        nG,
        source,
        end_time,
        perturbation,
        **kwargs,
    ):
        self.nx = nx
        self.end_time = end_time

        self.nG = nG
        self.category = category

        self.source = source
        self.perturbation = perturbation

        self.nQ = kwargs.get("nQ", self.nG + 1)

        self.BC = kwargs.get("BC", "exact")
        self.time_integrator = kwargs.get("time_integrator", "SSPRK34")
        self.cfl_factor = kwargs.get("cfl_factor", 1)

        if not self.source:
            self.BC = "periodic"

        assert self.BC in [
            "exact",
            "periodic",
        ], f"boundary conditions {self.BC} not implemented in DG_transport"

        assert self.category in [
            "no_prior",
            "with_prior_additive",
            "with_prior_multiplicative",
        ], f"category {self.category} not implemented in DG_transport"

        self.a_min = kwargs.get("a_min", self.DEFAULT_A_MIN)
        self.a_max = kwargs.get("a_max", self.DEFAULT_A_MAX)
        self.b_min = kwargs.get("b_min", self.DEFAULT_B_MIN)
        self.b_max = kwargs.get("b_max", self.DEFAULT_B_MAX)
        self.u0_min = kwargs.get("u0_min", self.DEFAULT_U0_MIN)
        self.u0_max = kwargs.get("u0_max", self.DEFAULT_U0_MAX)

        self.random_parameters = kwargs.get("random_parameters", False)

        self.fixed_parameters = kwargs.get("fixed_parameters", False)
        self.fixed_a = kwargs.get("fixed_a", None)
        self.fixed_b = kwargs.get("fixed_b", None)
        self.fixed_u0 = kwargs.get("fixed_u0", None)

        if "with_prior" in self.category:
            self.PINN = Network()

        try:
            self.time_integrator = kwargs.get(
                "time_integrator", Mesh.time_integrators[self.nG]
            )
        except KeyError:
            self.time_integrator = "SSPRK410"

        if "with_prior" in self.category:
            self.nQ = max(self.nG + 1, 3)

        self.dx = 1 / self.nx
        self.x = torch.linspace(0 + self.dx / 2, 1 - self.dx / 2, self.nx)

        self.c = 1

        if self.fixed_parameters:
            self.a = self.fixed_a
            self.b = self.fixed_b
            self.u0 = self.fixed_u0
        elif self.random_parameters:
            self.a = torch.random.uniform(self.a_min, self.a_max)
            self.b = torch.random.uniform(self.b_min, self.b_max)
            self.u0 = torch.random.uniform(self.u0_min, self.u0_max)
        else:
            self.a = (self.a_min + self.a_max) / 2
            self.b = (self.b_min + self.b_max) / 2
            self.u0 = (self.u0_min + self.u0_max) / 2

        self.p_Gauss, self.w_Gauss = DG_scheme.get_Gauss(self.nQ)

        self.dof = (
            self.x.repeat_interleave(self.nQ)
            + (torch.tile(self.p_Gauss, (self.nx,)) * 2 - 1) * self.dx / 2
        )
        self.nb_dof = len(self.dof)

        self.weights = torch.tile(self.w_Gauss, (self.nx,))

        self.dt = self.dx / 10

        self.dt *= Mesh.cfl_factors[self.time_integrator]
        self.dt *= self.cfl_factor

        self.t = torch.linspace(0, self.end_time, 2 + int(self.end_time / self.dt))
        self.nb_iter = len(self.t) - 1
        self.dt = self.t[1] - self.t[0]

        self.p_Gauss_loc = (
            self.x[:, None] + self.dx * self.p_Gauss[None, :] - self.dx / 2
        )

        self.phi = torch.zeros((self.nx, self.nG, self.nQ))
        self.d_phi = torch.zeros((self.nx, self.nG, self.nQ))

        for iG in range(self.nG):
            self.phi[:, iG] = self.phi_k(self.p_Gauss_loc, self.x[:, None], iG)
            self.d_phi[:, iG] = self.d_phi_k(self.p_Gauss_loc, self.x[:, None], iG)

        M_blocks = self.dx * torch.sum(
            self.w_Gauss * self.phi[:, :, None] * self.phi[:, None, :],
            axis=-1,
        )

        id = torch.eye(self.nG)[None, ...].repeat(M_blocks.shape[0], 1, 1)
        self.M_blocks_inv = torch.linalg.inv(1e-15 * id + M_blocks)

        if self.perturbation:
            self.errors_over_time = torch.zeros(self.nb_iter + 1)

    def initial_perturbation(self, x):
        return 1 + self.perturbation * torch.sin(2 * torch.pi * x)

    def phi_0(self, x: torch.Tensor, x0) -> torch.Tensor:
        """Compute the first basis function.

        Args:
            x: the points where to evaluate the basis function,

        Return:
            value of the first basis function at each point x
        """
        if self.category == "no_prior":
            return torch.ones_like(x)
        elif "with_prior" in self.category:
            return self.PINN.predict_u(x, self).detach()
        else:
            raise ValueError(f"category {self.category} not understood in phi_0")

    def d_phi_0(self, x: torch.Tensor, x0) -> torch.Tensor:
        """Compute the derivative of the first basis function.

        Args:
            x: the points where to evaluate the basis function,

        Return:
            value of the derivative of the first basis function at each point x
        """
        if self.category == "no_prior":
            return torch.zeros_like(x)
        elif "with_prior" in self.category:
            return self.PINN.predict_dxu(x, self).detach()
        else:
            raise ValueError(f"category {self.category} not understood in d_phi_0")

    def phi_k(self, x: torch.Tensor, x0: torch.Tensor, k: int) -> torch.Tensor:
        """Compute the k-th basis function.

        Args:
            x: the points where to evaluate the basis function,
            x0: the center of the cell where the basis function is evaluated
            k: the index of the basis function

        Return:
            value of the k-th basis function at each point x
        """
        if k == 0:
            return self.phi_0(x, x0)
        else:
            x_c = x - x0
            if "with_prior" in self.category:
                if self.category == "with_prior_multiplicative":
                    return x_c**k / k * self.phi_0(x, x0)
                elif self.category == "with_prior_additive":
                    return x_c**k / k
            else:
                return x_c**k / k

    def d_phi_k(self, x: torch.Tensor, x0: torch.Tensor, k: int) -> torch.Tensor:
        """Compute the derivative of the k-th basis function.

        Args:
            x: the points where to evaluate the basis function,
            x0: the center of the cell where the basis function is evaluated
            k: the index of the basis function

        Return:
            value of the derivative of the k-th basis function at each point x
        """
        if k == 0:
            return self.d_phi_0(x, x0)
        else:
            x_c = x - x0
            if "with_prior" in self.category:
                if self.category == "with_prior_multiplicative":
                    phi = self.phi_0(x, x0)
                    return (x_c ** (k - 1)) * phi + x_c**k / k * self.d_phi_0(x, x0)
                elif self.category == "with_prior_additive":
                    return x_c ** (k - 1)
            else:
                return x_c ** (k - 1)


def solve(category, nx, nG, source, end_time, perturbation):
    start = time.time()
    M = Mesh(
        category,
        nx,
        nG,
        source,
        end_time,
        perturbation,
    )
    M.cpu_time_mesh = time.time() - start

    start = time.time()
    W = DG_scheme.run(M)
    M.cpu_time_scheme = time.time() - start

    return W, M


def run_perturbation_analysis(categories, **kwargs):
    nx = kwargs.get("nx", 20)
    nG = kwargs.get("nG", 2)
    source = kwargs.get("source", True)
    end_time = kwargs.get("end_time", 2.5)

    perturbations = [1e-0, 1e-2, 1e-4]

    len(categories)
    n_p = len(perturbations)

    fig, ax = plt.subplots(1, n_p, figsize=(5 * n_p, 5))

    for i_p, perturbation in enumerate(perturbations):
        for i_c, category in enumerate(categories):
            W, M = solve(
                category,
                nx,
                nG,
                source,
                end_time,
                perturbation,
            )

            ax[i_p].semilogy(
                M.t.detach().cpu(), M.errors_over_time.detach().cpu(), label=category
            )

        ax[i_p].set_title(f"initial perturbation = {perturbation}")
        ax[i_p].legend()


def run_and_plot(categories, **kwargs):
    nx = kwargs.get("nx", 20)
    nG = kwargs.get("nG", 2)
    source = kwargs.get("source", True)
    end_time = kwargs.get("end_time", 0.5)

    perturbation = 0

    for category in categories:
        W, M = solve(
            category,
            nx,
            nG,
            source,
            end_time,
            perturbation,
        )
        DG_scheme.plot_and_compute_error(W, M)

