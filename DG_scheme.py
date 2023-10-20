"""
@author: Victor Michel-Dansac <victor.michel-dansac@inria.fr>
"""

import matplotlib.pyplot as plt
import numpy as np
import u_solutions


def get_Gauss(nQ: int) -> (np.array, np.array):
    if nQ == 2:
        p_Gauss = np.array([0.0, 1.0])
        w_Gauss = np.array([1 / 2, 1 / 2])

    elif nQ == 3:
        p_Gauss = np.array([0.0, 0.5, 1.0])
        w_Gauss = np.array([1 / 6, 2 / 3, 1 / 6])

    elif nQ == 4:
        p_Gauss = np.array([0, (5 - np.sqrt(5)) / 10, (5 + np.sqrt(5)) / 10, 1])
        w_Gauss = np.array([1, 5, 5, 1]) / 12

    elif nQ == 5:
        p_Gauss = np.array(
            [
                0,
                (np.sqrt(7) - np.sqrt(3)) / (2 * np.sqrt(7)),
                0.5,
                (np.sqrt(7) + np.sqrt(3)) / (2 * np.sqrt(7)),
                1,
            ]
        )
        w_Gauss = np.array([9, 49, 64, 49, 9]) / 180

    elif nQ == 6:
        p_Gauss = (
            1
            + np.array(
                [
                    -1,
                    -np.sqrt(1 / 3 + 2 * np.sqrt(7) / 21),
                    -np.sqrt(1 / 3 - 2 * np.sqrt(7) / 21),
                    np.sqrt(1 / 3 - 2 * np.sqrt(7) / 21),
                    np.sqrt(1 / 3 + 2 * np.sqrt(7) / 21),
                    1,
                ]
            )
        ) / 2
        w_Gauss = (
            np.array(
                [
                    2,
                    14 - np.sqrt(7),
                    14 + np.sqrt(7),
                    14 + np.sqrt(7),
                    14 - np.sqrt(7),
                    2,
                ]
            )
            / 60
        )

    else:
        raise ValueError(f"unrecognized value of nQ ({nQ}) in get_Gauss")

    return p_Gauss, w_Gauss


def physical_flux(u, c):
    return c * u


def upwind_flux(u_l, u_r, c):
    centred_term = (physical_flux(u_l, c) + physical_flux(u_r, c)) / 2
    diffusion = np.abs(c) * (u_r - u_l) / 2
    return centred_term - diffusion


def project_on_fine_mesh(u, x, mesh):
    nx = len(x)
    uh = np.zeros((nx,))
    mesh_indices = np.argmin(np.abs(x[:, None] - mesh.x[None, :]), axis=1)
    for j in range(mesh.nG):
        uh += u[mesh_indices * mesh.nG + j] * mesh.phi_k(x, mesh.x[mesh_indices], j)
    return uh


def compute_error(u, mesh):
    u_ = np.sum(u.reshape((mesh.nx, mesh.nG))[:, :, None] * mesh.phi, axis=1)

    difference = (u_ - u_solutions.u_exact(mesh.p_Gauss_loc, mesh.end_time, mesh)) ** 2

    return np.sqrt(mesh.dx * np.sum(difference))


def plot_and_compute_error(u, mesh):
    error = compute_error(u, mesh)

    n_visu = 1000
    x = np.linspace(0, 1, n_visu + 1)

    u_fine = project_on_fine_mesh(u, x, mesh)

    _, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].plot(x, u_fine, label="approx")
    ax[0].plot(x, u_solutions.u_exact(x, mesh.end_time, mesh), label="exact")

    ax[0].set_title(f"{mesh.category}")
    ax[0].legend()

    ax[1].plot(
        x, u_fine - u_solutions.u_exact(x, mesh.end_time, mesh), label="difference"
    )
    ax[1].legend()

    print(f"{mesh.category: >35s},  error = {error: 6.3e}")


def time_integration(W_0, mesh, iter):
    if mesh.time_integrator == "Explicit Euler":
        return W_0 + update(W_0, mesh, iter)

    elif mesh.time_integrator == "SSPRK2":
        W_1 = W_0 + update(W_0, mesh, iter)
        return (W_0 + W_1 + update(W_1, mesh, iter)) / 2

    elif mesh.time_integrator == "SSPRK3":
        W_1 = W_0 + update(W_0, mesh, iter)
        W_2 = (3 * W_0 + W_1 + update(W_1, mesh, iter)) / 4
        return (W_0 + 2 * W_2 + update(W_2, mesh, iter)) / 3

    elif mesh.time_integrator == "SSPRK34":
        W_1 = W_0 + update(W_0, mesh, iter) / 2
        W_2 = W_1 + update(W_1, mesh, iter) / 2
        W_3 = (4 * W_0 + 2 * W_2 + update(W_2, mesh, iter)) / 6
        return W_3 + update(W_3, mesh, iter) / 2

    elif mesh.time_integrator == "SSPRK35":
        a_10_L = 0.37726891511710
        a_21_L = 0.37726891511710
        a_30 = 0.56656131914033
        a_32 = 0.43343868085967
        a_32_L = 0.16352294089771
        a_40 = 0.09299483444413
        a_41 = 0.00002090369620
        a_43 = 0.90698426185967
        a_40_L = 0.00071997378654
        a_43_L = 0.34217696850008
        a_50 = 0.00736132260920
        a_51 = 0.20127980325145
        a_52 = 0.00182955389682
        a_54 = 0.78952932024253
        a_50_L = 0.00277719819460
        a_51_L = 0.00001567934613
        a_54_L = 0.29786487010104

        W_0_L = update(W_0, mesh, iter)
        W_1 = W_0 + a_10_L * W_0_L
        W_1_L = update(W_1, mesh, iter)
        W_2 = W_1 + a_21_L * W_1_L
        W_2_L = update(W_2, mesh, iter)
        W_3 = a_30 * W_0 + a_32 * W_2 + a_32_L * W_2_L
        W_3_L = update(W_3, mesh, iter)
        W_4 = a_40 * W_0 + a_41 * W_1 + a_43 * W_3 + a_40_L * W_0_L + a_43_L * W_3_L
        W_4_L = update(W_4, mesh, iter)
        return (
            a_50 * W_0
            + a_51 * W_1
            + a_52 * W_2
            + a_54 * W_4
            + a_50_L * W_0_L
            + a_51_L * W_1_L
            + a_54_L * W_4_L
        )

    elif mesh.time_integrator == "SSPRK45":
        a_01 = 0.391752226571890
        a_20 = 0.444370493651235
        a_21 = 0.555629506348765
        a_22 = 0.368410593050371
        a_30 = 0.620101851488403
        a_32 = 0.379898148511597
        a_33 = 0.251891774271694
        a_40 = 0.178079954393132
        a_43 = 0.821920045606868
        a_44 = 0.544974750228521
        a_52 = 0.517231671970585
        a_53 = 0.096059710526147
        a_53_L = 0.063692468666290
        a_54 = 0.386708617503269
        a_54_L = 0.226007483236906

        W_1_L = update(W_0, mesh, iter)
        W_1 = W_0 + a_01 * W_1_L

        W_2_L = update(W_1, mesh, iter)
        W_2 = a_20 * W_0 + a_21 * W_1 + a_22 * W_2_L

        W_3_L = update(W_2, mesh, iter)
        W_3 = a_30 * W_0 + a_32 * W_2 + a_33 * W_3_L

        W_4_L = update(W_3, mesh, iter)
        W_4 = a_40 * W_0 + a_43 * W_3 + a_44 * W_4_L

        W_5_L = update(W_4, mesh, iter)
        W_5 = a_52 * W_2 + a_53 * W_3 + a_53_L * W_4_L + a_54 * W_4 + a_54_L * W_5_L

        return W_5

    elif mesh.time_integrator == "SSPRK410":
        W_1 = np.copy(W_0)
        for _ in range(5):
            W_1 += update(W_1, mesh, iter) / 6
        W_2 = 1 / 25 * W_0 + 9 / 25 * W_1
        W_1 = 15 * W_2 - 5 * W_1
        for _ in range(4):
            W_1 += update(W_1, mesh, iter) / 6
        return W_2 + 3 / 5 * W_1 + 1 / 10 * update(W_1, mesh, iter)


def initialize(M):
    u_0 = u_solutions.u_ini(M.p_Gauss_loc, M)

    if M.perturbation:
        u_0 *= M.initial_perturbation(M.p_Gauss_loc)

    u_0_ = M.w_Gauss * M.phi * u_0[:, None, :]
    rhs = M.dx * np.sum(u_0_, axis=-1).reshape(M.nx * M.nG)

    return M.inv_mass_matrix.solve(rhs)


def run(M):
    iter = 0
    W = initialize(M)

    while iter < M.nb_iter:
        W = time_integration(W, M, iter)

        if M.perturbation:
            error = compute_error(W, M)
            M.errors_over_time[iter] = error

        iter += 1

    if M.perturbation:
        error = compute_error(W, M)
        M.errors_over_time[iter] = error

    return W


def compute_BC(u_kmh, u_kph, mesh, iter):
    assert mesh.BC in ["periodic", "exact"]

    if mesh.BC == "periodic":
        return np.roll(u_kph, +1), np.roll(u_kmh, -1)

    elif mesh.BC == "exact":
        u_kph_km1 = np.zeros_like(u_kph)
        u_kmh_kp1 = np.zeros_like(u_kmh)

        u_kph_km1[+1:] = u_kph[:-1]
        u_kmh_kp1[:-1] = u_kmh[+1:]

        u_kph_km1[:+1] = u_solutions.u_exact(mesh.dof[+0], mesh.dt * iter, mesh)
        u_kmh_kp1[-1:] = u_solutions.u_exact(mesh.dof[-1], mesh.dt * iter, mesh)

        return u_kph_km1, u_kmh_kp1


def update(W, mesh, iter):
    # get local variables from mesh

    nx, nG = mesh.nx, mesh.nG
    c, dx, dt = mesh.c, mesh.dx, mesh.dt

    phi, d_phi = mesh.phi, mesh.d_phi

    v_shape = nx * nG
    m_shape = (nx, nG)

    # mass matrix

    inv_M = mesh.inv_mass_matrix

    # volume term

    u = np.sum(W.reshape(m_shape)[:, :, None] * phi, axis=1)
    K = dx * np.sum(
        mesh.w_Gauss * physical_flux(u, c)[:, None, :] * d_phi, axis=-1
    ).reshape(v_shape)

    # flux term

    u_kmh = np.sum(W.reshape(m_shape) * phi[:, :, +0], axis=1)
    u_kph = np.sum(W.reshape(m_shape) * phi[:, :, -1], axis=1)

    u_kph_km1, u_kmh_kp1 = compute_BC(u_kmh, u_kph, mesh, iter)

    F = (
        upwind_flux(u_kph, u_kmh_kp1, mesh.c)[:, None] * phi[:, :, -1]
        - upwind_flux(u_kph_km1, u_kmh, mesh.c)[:, None] * phi[:, :, +0]
    ).reshape(v_shape)

    flux_source = K - F

    # source term

    if mesh.source:
        S = mesh.a * u + mesh.b * u**2
        source_integral = dx * np.sum(
            mesh.w_Gauss * S[:, None, :] * phi, axis=-1
        ).reshape(v_shape)

        flux_source += source_integral

    flux_source = inv_M.solve(flux_source)

    return dt * flux_source
