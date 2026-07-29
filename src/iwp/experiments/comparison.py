"""Shared setup for the Part 4 (Distributed Optimization) / Part 5 (projected
Chambolle-Pock) comparison against the baselines of the original internship
report (P-ClosedForm, C-GD/C-NAGD/C-SCNAGD, FB/FISTA).

This module intentionally factors out the ~150 lines of boilerplate that
`main.py`, `experiment.ipynb` and `exp_dbgd.ipynb` each redefine inline
(loading the data, assembling the stacked `D`/`E` operators, the `J_1`/`J_2`/
`J_3` objective/gradient/prox factories) so that both the comparison script
(`scripts/compare_algorithms.py`) and the comparison notebook build on a
single, tested implementation instead of three divergent copies.
"""

import os
import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from iwp.algorithms.algorithms import AffineConstraintProjector
from iwp.data.export import export_all_metrics_to_csv, save_complex_vector
from iwp.data.load_experiment_data import load_experiment_data
from iwp.utils.operators import (
    build_graph_gradient_from_B,
    power_iteration_operator_norm,
)

# ---------------------------------------------------------------------------
# Problem setup
# ---------------------------------------------------------------------------


@dataclass
class ProblemData:
    A: sp.spmatrix
    B_list: list
    C: sp.spmatrix
    d_list: list
    m: np.ndarray
    I: int
    J: int
    L: int
    P: int
    A_star: sp.spmatrix = field(repr=False)
    C_star: sp.spmatrix = field(repr=False)
    D: sp.spmatrix = field(repr=False)  # stacked data operator, Eq. (6)
    D_star: sp.spmatrix = field(repr=False)
    E: sp.spmatrix = field(repr=False)  # stacked PDE operator, Eq. (26)/(47)
    E_star: sp.spmatrix = field(repr=False)
    d: np.ndarray = field(repr=False)  # concatenated d_list
    G: sp.spmatrix = field(repr=False)  # graph-gradient TV proxy, Sec. 5.2


def load_problem(data_path: str) -> ProblemData:
    """Load a FreeFEM-exported dataset and assemble every stacked operator
    needed by both the baselines of the internship report (`D`, `E`) and the
    new Part 4/5 algorithms (the same `E`, plus a graph-gradient TV proxy
    `G` built from the `B_i` sparsity pattern, cf.
    `iwp.utils.operators.build_graph_gradient_from_B`)."""
    A, B_list, C, d_list, m = load_experiment_data(data_path)
    A_star = A.conj().T
    C_star = C.conj().T

    I = len(B_list)
    J, L = C.shape
    P = B_list[0].shape[1]

    row_blocks = []
    for i in range(I):
        blocks = [sp.csr_matrix((J, L))] * I + [sp.csr_matrix((J, P))]
        blocks[i] = sp.csr_matrix(C)
        row_blocks.append(sp.hstack(blocks, format="csr"))
    D = sp.vstack(row_blocks, format="csr")
    D_star = D.conj().T

    d = np.concatenate(d_list, axis=0)

    row_blocks = []
    for i in range(I):
        blocks = [sp.csr_matrix((L, L))] * I + [-B_list[i]]
        blocks[i] = sp.csr_matrix(A)
        row_blocks.append(sp.hstack(blocks, format="csr"))
    E = sp.vstack(row_blocks, format="csr")
    E_star = E.conj().T

    G = build_graph_gradient_from_B(B_list)

    return ProblemData(
        A=A,
        B_list=B_list,
        C=C,
        d_list=d_list,
        m=m,
        I=I,
        J=J,
        L=L,
        P=P,
        A_star=A_star,
        C_star=C_star,
        D=D,
        D_star=D_star,
        E=E,
        E_star=E_star,
        d=d,
        G=G,
    )


# ---------------------------------------------------------------------------
# Baseline objective/gradient/prox factories, reproduced faithfully from
# main.py / experiment.ipynb (P-ClosedForm <- J_1, FISTA/FB <- J_2,
# C-GD/C-NAGD/C-SCNAGD <- J_3). Kept as free functions taking a `ProblemData`
# so both the script and the notebook get byte-identical baselines.
# ---------------------------------------------------------------------------


def get_J_1(pb: ProblemData, lambd, mu):
    D, E, d = pb.D, pb.E, pb.d

    def J_1(x):
        Dx_minus_d = D @ x - d
        Ex = E @ x
        return (
            0.5 * np.vdot(Dx_minus_d, Dx_minus_d).real
            + 0.5 * lambd * np.vdot(Ex, Ex).real
            + 0.5 * mu * np.vdot(x, x).real
        )

    return J_1


def get_dJ_1(pb: ProblemData, lambd, mu):
    D, D_star, E, E_star, d = pb.D, pb.D_star, pb.E, pb.E_star, pb.d

    def dJ_1(x):
        Dx_minus_d = D @ x - d
        Ex = E @ x
        return D_star @ Dx_minus_d + lambd * (E_star @ Ex) + mu * x

    return dJ_1


def get_closed_form_solution_J_1(pb: ProblemData, lambd, mu):
    D, D_star, E_star, E = pb.D, pb.D_star, pb.E_star, pb.E

    def closed_form_solution_J_1():
        return sp.linalg.spsolve(
            D_star @ D + lambd * (E_star @ E) + mu * sp.eye(pb.I * pb.L + pb.P),
            D_star @ pb.d,
        )

    return closed_form_solution_J_1


def get_K_J_1(pb: ProblemData, lambd, mu):
    D, D_star, E, E_star = pb.D, pb.D_star, pb.E, pb.E_star
    K_op = D_star @ D + lambd * (E_star @ E) + mu * sp.eye(pb.I * pb.L + pb.P)
    return float(np.max(np.abs(np.linalg.eigvals(K_op.toarray()))))


def get_J_2(pb: ProblemData, mu, threshold=1e-6):
    D, E, d, P = pb.D, pb.E, pb.d, pb.P

    def J_2(x):
        Dx_minus_d = D @ x - d
        Ex = E @ x
        if np.linalg.norm(Ex) < threshold:
            return (
                0.5 * np.vdot(Dx_minus_d, Dx_minus_d).real
                + 0.5 * mu * np.vdot(x[-P:], x[-P:]).real
            )
        return np.inf

    return J_2


def get_grad_J_2(pb: ProblemData, mu):
    D_star, D, d, P = pb.D_star, pb.D, pb.d, pb.P

    def grad_J_2(x):
        reg = np.zeros_like(x)
        reg[-P:] = mu * x[-P:]
        return D_star @ (D @ x - d) + reg

    return grad_J_2


def get_prox_J_2_spsolve(pb: ProblemData):
    """Naive projection prox: re-forms and re-solves `(E E*) w = E x` from
    scratch on every call -- exactly the original report's implementation,
    i.e. projector backend "S1" of `AffineConstraintProjector`."""
    E, E_star = pb.E, pb.E_star

    def prox_J_2(x, gamma):
        w = sp.linalg.spsolve(E @ E_star, E @ x)
        return x - E_star @ w

    return prox_J_2


def make_prox_J_2_from_projector(pb: ProblemData, projector: AffineConstraintProjector):
    """Same forward-backward/FISTA projection step as `get_prox_J_2_spsolve`,
    but delegated to an `AffineConstraintProjector` (S2/S3/S4 backends),
    letting the *same* FISTA algorithm be timed with different projection
    implementations for a fair backend-only comparison (Sec. 5.8-5.9)."""
    I, L, P = pb.I, pb.L, pb.P

    def prox_J_2(x, gamma):
        u_list = [x[i * L : (i + 1) * L] for i in range(I)]
        m = x[I * L : I * L + P]
        u_new, m_new = projector.project(u_list, m)
        out = np.empty_like(x)
        for i in range(I):
            out[i * L : (i + 1) * L] = u_new[i]
        out[I * L : I * L + P] = m_new
        return out

    return prox_J_2


def get_K_J_2(pb: ProblemData, mu):
    D_star, D = pb.D_star, pb.D
    K_op = D_star @ D + mu * sp.eye(pb.I * pb.L + pb.P)
    return float(np.max(np.abs(np.linalg.eigvals(K_op.toarray()))))


def get_J_3(pb: ProblemData, mu):
    A, C, B_list, d_list = pb.A, pb.C, pb.B_list, pb.d_list

    def J_3(m):
        total = 0.0
        for i in range(len(B_list)):
            CA_inv_Bi_m = C @ sp.linalg.spsolve(A, B_list[i] @ m)
            diff = CA_inv_Bi_m - d_list[i]
            total += 0.5 * np.vdot(diff, diff).real
        return total + 0.5 * mu * np.vdot(m, m).real

    return J_3


def get_dJ_3(pb: ProblemData, mu):
    A, A_star, C, C_star, B_list, d_list = (
        pb.A,
        pb.A_star,
        pb.C,
        pb.C_star,
        pb.B_list,
        pb.d_list,
    )

    def dJ_3(m):
        p_sum = sum(
            B_i.conj().T
            @ sp.linalg.spsolve(
                A_star, C_star @ (C @ sp.linalg.spsolve(A, B_i @ m) - d_i)
            )
            for B_i, d_i in zip(B_list, d_list)
        )
        return p_sum + mu * m

    return dJ_3


def get_K_J_3(pb: ProblemData, mu):
    A, C, B_list = pb.A, pb.C, pb.B_list
    Ainv = sp.linalg.inv(A.tocsc())
    K_op = sum(
        B_i.conj().T @ Ainv.conj().T @ C.conj().T @ C @ Ainv @ B_i for B_i in B_list
    ) + mu * sp.eye(pb.P)
    return float(np.max(np.abs(np.linalg.eigvals(K_op.toarray()))))


# ---------------------------------------------------------------------------
# Operator-norm estimation for the Chambolle-Pock family (Sec. 4.7, 5.2, 5.6).
# Power iteration is used throughout so these remain tractable when I grows
# in the source-count sweep (dense eigvals on the I*L-scale operators used
# above for J_1/J_2 would become prohibitive; those are only ever evaluated
# at the small, fixed I=2 baseline dataset).
# ---------------------------------------------------------------------------


def l_operator_norm_algorithm3(pb: ProblemData, G=None, n_iter=200, seed=0):
    """Operator norm of `L x = ((A u_i - B_i m)_i, G m)` (Eq. (26)), the
    operator dualized by `ChambollePock`/`DistributedChambollePock`
    (Algorithm 3/4). Estimated by power iteration (Sec. 4.7: "in practice we
    estimate ||L|| by a few power iterations on L*L") rather than the a
    priori bound of Eq. (33), which is intentionally loose."""
    A, A_star, B, B_star = pb.A, pb.A_star, pb.B_list, [Bi.conj().T for Bi in pb.B_list]
    I, L, P = pb.I, pb.L, pb.P
    Q = G.shape[0] if G is not None else 0
    G_star = G.conj().T if G is not None else None

    def matvec(x):
        u = [x[i * L : (i + 1) * L] for i in range(I)]
        m = x[I * L : I * L + P]
        out = np.concatenate([A @ u[i] - B[i] @ m for i in range(I)])
        if G is not None:
            out = np.concatenate([out, G @ m])
        return out

    def rmatvec(y):
        v_pde = [y[i * L : (i + 1) * L] for i in range(I)]
        out_u = np.concatenate([A_star @ v_pde[i] for i in range(I)])
        out_m = -sum(B_star[i] @ v_pde[i] for i in range(I))
        if G is not None:
            v_tv = y[I * L : I * L + Q]
            out_m = out_m + G_star @ v_tv
        return np.concatenate([out_u, out_m])

    return power_iteration_operator_norm(
        matvec, rmatvec, dim=I * L + P, n_iter=n_iter, seed=seed
    )


def k_operator_norm_algorithm5(pb: ProblemData, G=None, n_iter=200, seed=0):
    """Operator norm of `K x = ((C u_i)_i, G m)` (Eq. (43)), the operator
    dualized by `ProjectedChambollePock` (Algorithm 5). Crucially never
    involves `A`, unlike `l_operator_norm_algorithm3` above -- this is the
    structural mesh-robustness advantage motivating Sec. 5."""
    C, C_star = pb.C, pb.C_star
    I, L, P, J = pb.I, pb.L, pb.P, pb.J
    Q = G.shape[0] if G is not None else 0
    G_star = G.conj().T if G is not None else None

    def matvec(x):
        u = [x[i * L : (i + 1) * L] for i in range(I)]
        m = x[I * L : I * L + P]
        out = np.concatenate([C @ u[i] for i in range(I)])
        if G is not None:
            out = np.concatenate([out, G @ m])
        return out

    def rmatvec(y):
        v_dat = [y[i * J : (i + 1) * J] for i in range(I)]
        out_u = np.concatenate([C_star @ v_dat[i] for i in range(I)])
        if G is not None:
            v_tv = y[I * J : I * J + Q]
            out_m = G_star @ v_tv
        else:
            out_m = np.zeros(P, dtype=complex)
        return np.concatenate([out_u, out_m])

    return power_iteration_operator_norm(
        matvec, rmatvec, dim=I * L + P, n_iter=n_iter, seed=seed
    )


# ---------------------------------------------------------------------------
# Block-preconditioned dual step sizes (Eq. (34) for Algorithm 3/4, Eq. (56)
# for Algorithm 5).
#
# Both dualized operators stack two blocks whose scales differ by orders of
# magnitude, so a single scalar `sigma` is throttled by the larger of the two
# and wastes the freedom of the smaller:
#
#   Algorithm 3/4:  L x = ( (A u_i - B_i m)_i , G m )   -- ||A|| = O(1/h^2)
#                                                          vs ||G|| = O(1/h)
#   Algorithm 5:    K x = ( (C u_i)_i         , G m )   -- ||C|| small
#                                                          vs ||G|| = O(1/h)
#
# Giving each block its own dual step through the metric
# `Sigma = diag(sigma_pde/dat I, sigma_reg I)` relaxes the convergence
# condition from `tau sigma ||.||^2 < 1` to `tau ||Sigma^(1/2) . ||^2 < 1`.
# The helpers below implement the recipe of Sec. 5.6 verbatim: estimate each
# block norm *separately* by power iteration (never through the a priori
# bounds (33)/(46), which are pessimistic), set `sigma_block = gamma /
# ||block||^2`, then rescale `tau` so that the left-hand side of the condition
# equals `safety` (< 1). This leaves a single scalar knob `gamma`, the
# primal/dual step ratio, to be tuned by residual balancing.
# ---------------------------------------------------------------------------


def _weighted_norm(matvec, rmatvec, dim, n_iter=200, seed=0):
    """`||Sigma^(1/2) K||`, with the weighting already folded into the two
    callables (`matvec = Sigma^(1/2) K`, `rmatvec = K* Sigma^(1/2)`)."""
    return power_iteration_operator_norm(
        matvec, rmatvec, dim=dim, n_iter=n_iter, seed=seed
    )


def block_step_sizes_algorithm3(
    pb: ProblemData, G=None, gamma=1.0, safety=0.9, n_iter=200, seed=0
):
    """Block-preconditioned `(tau, sigma_pde, sigma_reg)` for `ChambollePock`
    / `DistributedChambollePock` (Algorithm 3/4) under the metric
    `Sigma = diag(sigma_pde I, sigma_reg I)` and the condition (Eq. (34))

        tau * ||Sigma^(1/2) L||^2 < 1.

    `gamma` is the single remaining knob (the primal/dual ratio): larger
    `gamma` means larger dual steps and, through the rescaling of `tau`, a
    smaller primal step. `gamma = ||L||` reproduces the balanced scalar rule
    `tau = sigma = sqrt(safety)/||L||` used elsewhere in this module, so
    sweeping `gamma` around that value is the natural way to compare the two
    metrics at fixed convergence margin. `G=None` disables the regularizer
    block, in which case `sigma_reg` is returned as `None` and the result
    reduces to the scalar rule up to the `gamma` reparametrization.

    Returns a dict with the step sizes, the two block norms, the weighted
    norm `||Sigma^(1/2) L||`, and the realized left-hand side of Eq. (34).
    """
    A, A_star = pb.A, pb.A_star
    B, B_star = pb.B_list, [Bi.conj().T for Bi in pb.B_list]
    I, L, P = pb.I, pb.L, pb.P
    dim = I * L + P
    G_star = G.conj().T if G is not None else None

    def pde_matvec(x):
        u = [x[i * L : (i + 1) * L] for i in range(I)]
        m = x[I * L : I * L + P]
        return np.concatenate([A @ u[i] - B[i] @ m for i in range(I)])

    def pde_rmatvec(y):
        v = [y[i * L : (i + 1) * L] for i in range(I)]
        out_u = np.concatenate([A_star @ v[i] for i in range(I)])
        out_m = -sum(B_star[i] @ v[i] for i in range(I))
        return np.concatenate([out_u, out_m])

    norm_pde = power_iteration_operator_norm(
        pde_matvec, pde_rmatvec, dim=dim, n_iter=n_iter, seed=seed
    )
    sigma_pde = gamma / norm_pde**2

    if G is None:
        # Single block: ||Sigma^(1/2) L||^2 = sigma_pde ||L_pde||^2 exactly.
        weighted_sq = sigma_pde * norm_pde**2
        tau = safety / weighted_sq
        return dict(
            tau=tau,
            sigma_pde=sigma_pde,
            sigma_reg=None,
            norm_pde=norm_pde,
            norm_reg=None,
            weighted_norm=float(np.sqrt(weighted_sq)),
            condition_lhs=tau * weighted_sq,
            gamma=gamma,
        )

    def reg_matvec(x):
        return G @ x[I * L : I * L + P]

    def reg_rmatvec(y):
        return np.concatenate([np.zeros(I * L, dtype=complex), G_star @ y])

    norm_reg = power_iteration_operator_norm(
        reg_matvec, reg_rmatvec, dim=dim, n_iter=n_iter, seed=seed
    )
    sigma_reg = gamma / norm_reg**2

    # ||Sigma^(1/2) L|| by power iteration on the *weighted* operator, rather
    # than through the sufficient bound sigma_pde ||L_pde||^2 + sigma_reg
    # ||L_reg||^2 (= 2 gamma here), which ignores the two blocks' interaction.
    s_pde, s_reg = np.sqrt(sigma_pde), np.sqrt(sigma_reg)

    def matvec(x):
        return np.concatenate([s_pde * pde_matvec(x), s_reg * reg_matvec(x)])

    def rmatvec(y):
        return s_pde * pde_rmatvec(y[: I * L]) + s_reg * reg_rmatvec(y[I * L :])

    weighted_norm = _weighted_norm(matvec, rmatvec, dim=dim, n_iter=n_iter, seed=seed)
    tau = safety / weighted_norm**2
    return dict(
        tau=tau,
        sigma_pde=sigma_pde,
        sigma_reg=sigma_reg,
        norm_pde=norm_pde,
        norm_reg=norm_reg,
        weighted_norm=weighted_norm,
        condition_lhs=tau * weighted_norm**2,
        gamma=gamma,
    )


def block_step_sizes_algorithm5(
    pb: ProblemData, G=None, gamma=1.0, safety=0.9, n_iter=200, seed=0
):
    """Block-preconditioned `(tau, sigma_dat, sigma_reg)` for
    `ProjectedChambollePock` (Algorithm 5) under `tau ||Sigma^(1/2) K||^2 < 1`
    (Eq. (56)), with `K x = ((C u_i)_i, G m)` (Eq. (43)).

    Same recipe and same `gamma` knob as `block_step_sizes_algorithm3`. This
    is where per-block freedom actually pays off in full (Sec. 5.6): unlike
    Algorithm 3, whose PDE block keeps `||A||` inside `L` however the metric
    is balanced, here both block norms (`||C||` and `||G||`) are moderate, so
    `sigma_dat >> sigma_reg` genuinely lets the data dual advance quickly
    while the regularizer dual respects its `O(1/h)` Lipschitz limit.

    `G=None` (or `reg_mode='tikhonov'`, for which the regularizer block is the
    identity `P_m` -- pass `G=sp.eye(P)` to weight it separately) reduces to
    the single-block case.
    """
    C, C_star = pb.C, pb.C_star
    I, L, P, J = pb.I, pb.L, pb.P, pb.J
    dim = I * L + P
    G_star = G.conj().T if G is not None else None

    def dat_matvec(x):
        u = [x[i * L : (i + 1) * L] for i in range(I)]
        return np.concatenate([C @ u[i] for i in range(I)])

    def dat_rmatvec(y):
        v = [y[i * J : (i + 1) * J] for i in range(I)]
        out_u = np.concatenate([C_star @ v[i] for i in range(I)])
        return np.concatenate([out_u, np.zeros(P, dtype=complex)])

    norm_dat = power_iteration_operator_norm(
        dat_matvec, dat_rmatvec, dim=dim, n_iter=n_iter, seed=seed
    )
    sigma_dat = gamma / norm_dat**2

    if G is None:
        weighted_sq = sigma_dat * norm_dat**2
        tau = safety / weighted_sq
        return dict(
            tau=tau,
            sigma_dat=sigma_dat,
            sigma_reg=None,
            norm_dat=norm_dat,
            norm_reg=None,
            weighted_norm=float(np.sqrt(weighted_sq)),
            condition_lhs=tau * weighted_sq,
            gamma=gamma,
        )

    def reg_matvec(x):
        return G @ x[I * L : I * L + P]

    def reg_rmatvec(y):
        return np.concatenate([np.zeros(I * L, dtype=complex), G_star @ y])

    norm_reg = power_iteration_operator_norm(
        reg_matvec, reg_rmatvec, dim=dim, n_iter=n_iter, seed=seed
    )
    sigma_reg = gamma / norm_reg**2

    s_dat, s_reg = np.sqrt(sigma_dat), np.sqrt(sigma_reg)

    def matvec(x):
        return np.concatenate([s_dat * dat_matvec(x), s_reg * reg_matvec(x)])

    def rmatvec(y):
        return s_dat * dat_rmatvec(y[: I * J]) + s_reg * reg_rmatvec(y[I * J :])

    weighted_norm = _weighted_norm(matvec, rmatvec, dim=dim, n_iter=n_iter, seed=seed)
    tau = safety / weighted_norm**2
    return dict(
        tau=tau,
        sigma_dat=sigma_dat,
        sigma_reg=sigma_reg,
        norm_dat=norm_dat,
        norm_reg=norm_reg,
        weighted_norm=weighted_norm,
        condition_lhs=tau * weighted_norm**2,
        gamma=gamma,
    )


# ---------------------------------------------------------------------------
# Uniform run/plot/export helper, factoring the pattern repeated for every
# algorithm across main.py / experiment.ipynb / exp_dbgd.ipynb.
# ---------------------------------------------------------------------------


def run_and_record(
    algo,
    x0,
    max_iterations,
    m_true,
    visuals_path=None,
    results_path=None,
    show=False,
    save=None,
):
    """Run `algo`, plot its convergence against `m_true`, and (if paths are
    given) export its metrics/prediction to disk. Returns the final iterate.
    """
    save = save if save is not None else (visuals_path is not None)
    t0 = time.time()
    x_final = algo.run(x0=x0, max_iterations=max_iterations)
    wall_time = time.time() - t0
    algo.plot_algorithm_convergence(
        m_true, visuals_path, show=show, save=save and visuals_path is not None
    )
    if results_path is not None:
        export_all_metrics_to_csv(
            algo, os.path.join(results_path, f"{algo.algo_plot_name}_Metrics.csv")
        )
        P = m_true.shape[0]
        save_complex_vector(
            os.path.join(results_path, f"{algo.algo_plot_name}_PredictedVectorm.dat"),
            x_final[-P:],
        )
    return x_final, wall_time
