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

import logging
import os
import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from iwp.algorithms.algorithms import AffineConstraintProjector
from iwp.data.export import export_all_metrics_to_csv, save_complex_vector
from iwp.data.load_experiment_data import load_experiment_data
from iwp.utils.mesh import (
    build_fe_jump_operator,
    load_contrast_mesh,
    load_dims,
    load_mesh,
    validate_dof_ordering,
)
from iwp.utils.operators import (
    build_graph_gradient_from_B,
    power_iteration_operator_norm,
)

logger = logging.getLogger("iwp")

# Regularizer-operator modes accepted by `load_problem`/`build_regularizer_operator`.
G_MODES = ("proxy", "fe_tv", "fe_h1")

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
    G: sp.spmatrix = field(repr=False)  # regularizer operator, see `G_mode`
    # --- everything below is additive: the defaults reproduce the original
    # dataclass exactly, so `ProblemData(...)` calls written before the
    # finite-element jump operator existed keep working unchanged.
    G_mode: str = "proxy"
    # Field-mesh connectivity, when the dataset carries a `mesh.msh` export
    # (`savemesh` in scripts/GenerateMatrix*.edp); None for older datasets.
    vertices: np.ndarray = field(default=None, repr=False)
    triangles: np.ndarray = field(default=None, repr=False)
    # Contrast-mesh connectivity. Identical to the field mesh for every
    # single-mesh dataset, and the *coarse* mesh for one generated with
    # `-delta_m` (Sec. 5.8's two-basis discretization). This, not
    # `vertices`/`triangles`, is what has exactly `P` triangles, and what the
    # jump operator and any contrast-space plot must be built on.
    contrast_vertices: np.ndarray = field(default=None, repr=False)
    contrast_triangles: np.ndarray = field(default=None, repr=False)
    # True when the two meshes differ, i.e. the dataset is a two-basis one.
    decoupled_mesh: bool = False
    # Mesh densities as recorded by the generator (`dims.txt`), None for
    # datasets that predate it.
    delta: int = None
    delta_m: int = None
    # Discrete H^1 (TPFA) jump operator, always built when the mesh is
    # available. This, rather than the TV operator, is what a first-order
    # Tikhonov control must use: with `w_E = |E|` the "H^1" term would be a
    # length-weighted sum of squared jumps, which is not an H^1 seminorm.
    G_h1: sp.spmatrix = field(default=None, repr=False)


def build_regularizer_operator(data_path, B_list, G_mode="proxy", mesh=None):
    """Build the operator `G` entering the regularizer block of Algorithms
    3/4/5, in one of three modes:

    ``"proxy"``  (default, unchanged behaviour)
        The structural graph gradient inferred from the `B_i` sparsity
        pattern, `iwp.utils.operators.build_graph_gradient_from_B`. Declares
        two triangles adjacent whenever they share *any* P1 field dof, so it
        is a strict superset of edge adjacency (2202 rows for P=394 triangles
        at delta=10, against 566 true interior edges), and all its entries
        are +-1, so it carries no geometry at all.

    ``"fe_tv"``
        The true finite-element inter-element jump operator with the edge
        length weights `w_E = |E|`, i.e. the discrete total variation that is
        *exact* on DG0 (Herrmann et al. 2019, Cor. 3.5(a)). See
        `iwp.utils.mesh.build_fe_jump_operator`.

    ``"fe_h1"``
        Same connectivity, TPFA weights `w_E = sqrt(|E|/d_E)`, whose
        quadratic form is the discrete H^1 seminorm.

    Kept as a free function so a caller can build both operators for the same
    dataset and compare them side by side without reloading it.

    Args:
        mesh: optional `(vertices, triangles)` of the *contrast* mesh, already
            parsed and validated, to avoid re-reading it; `load_problem`
            passes its own. On a two-basis dataset this is the coarse mesh,
            not the field mesh `mesh.msh` holds.
    """
    if G_mode == "proxy":
        return build_graph_gradient_from_B(B_list)
    if G_mode in ("fe_tv", "fe_h1"):
        vertices, triangles = (
            mesh if mesh is not None else load_contrast_mesh(data_path)[:2]
        )
        mode = "tv" if G_mode == "fe_tv" else "h1"
        return build_fe_jump_operator(vertices, triangles, mode=mode)
    raise ValueError(f"Unknown G_mode: {G_mode!r} (expected one of {G_MODES})")


def load_problem(
    data_path: str, G_mode: str = "proxy", validate_mesh: bool = True
) -> ProblemData:
    """Load a FreeFEM-exported dataset and assemble every stacked operator
    needed by both the baselines of the internship report (`D`, `E`) and the
    new Part 4/5 algorithms (the same `E`, plus the regularizer operator `G`).

    Args:
        G_mode: which regularizer operator to put in `pb.G`. `"proxy"`
            (default, the historical graph-gradient proxy, so every number
            already published in the notebook reproduces bit for bit),
            `"fe_tv"` or `"fe_h1"`. See `build_regularizer_operator`.
        validate_mesh: when a mesh export is present, check FreeFEM's dof
            ordering against `P0barycenters.dat`/`P1vertices.dat` before
            trusting the connectivity. A mismatch means `G` would be a
            silently permuted operator, so it raises rather than warns.
    """
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

    # Mesh connectivity, when the dataset has it. Loaded regardless of
    # `G_mode` so that plotting on the real geometry and the H^1 control are
    # available even in the default (proxy) configuration; `pb.G` itself is
    # untouched by this.
    vertices = triangles = G_h1 = None
    contrast_vertices = contrast_triangles = None
    decoupled_mesh = False
    try:
        vertices, triangles, _, _ = load_mesh(data_path)
        contrast_vertices, contrast_triangles, _, _, decoupled_mesh = (
            load_contrast_mesh(data_path)
        )
    except FileNotFoundError:
        if G_mode != "proxy":
            raise
        logger.info(
            f"No mesh export in {data_path}; only G_mode='proxy' is available "
            "and spatial plots are disabled."
        )
    if vertices is not None:
        if validate_mesh:
            check = validate_dof_ordering(
                data_path,
                vertices,
                triangles,
                contrast_vertices=contrast_vertices,
                contrast_triangles=contrast_triangles,
            )
            if not check["ok"]:
                raise ValueError(
                    f"Mesh dof ordering mismatch in {data_path}: "
                    f"barycentre error {check['max_error_p0']:.3e}, vertex error "
                    f"{check['max_error_p1']:.3e} (tolerance 1e-10). The parsed "
                    "connectivity does not match FreeFEM's dof numbering, so any "
                    "jump operator built from it would be silently permuted."
                )
        # The contrast dofs are the triangles of the *contrast* mesh, which is
        # the field mesh only when the dataset is a single-mesh one.
        if contrast_triangles.shape[0] != P:
            raise ValueError(
                f"Contrast mesh in {data_path} has {contrast_triangles.shape[0]} "
                f"triangles but the contrast space has P={P} dofs: mesh and "
                "matrices disagree."
            )
        if vertices.shape[0] != L:
            raise ValueError(
                f"Field mesh in {data_path} has {vertices.shape[0]} vertices but "
                f"the field space has L={L} dofs: mesh and matrices disagree."
            )
        G_h1 = build_fe_jump_operator(contrast_vertices, contrast_triangles, mode="h1")

    G = (
        G_h1
        if G_mode == "fe_h1" and G_h1 is not None
        else build_regularizer_operator(
            data_path, B_list, G_mode,
            mesh=None
            if contrast_vertices is None
            else (contrast_vertices, contrast_triangles),
        )
    )

    dims = load_dims(data_path)

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
        G_mode=G_mode,
        vertices=vertices,
        triangles=triangles,
        contrast_vertices=contrast_vertices,
        contrast_triangles=contrast_triangles,
        decoupled_mesh=decoupled_mesh,
        delta=None if dims is None else dims.get("delta"),
        delta_m=None if dims is None else dims.get("delta_m"),
        G_h1=G_h1,
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
    scratch on every call, which is exactly the original report's
    implementation, i.e. projector backend "S1" of `AffineConstraintProjector`."""
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
    involves `A`, unlike `l_operator_norm_algorithm3` above. This is the
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


def block_step_sizes_algorithm5(pb, G, tau=None, theta=0.81, n_iter=200, seed=0):
    """Block-preconditioned step sizes for `ProjectedChambollePock`
    (Eq. (56)): one dual step per block of `K x = ((C u_i)_i, G m)` instead of
    the single scalar `sigma = 0.9/||K||` of Eq. (33).

    The two dual blocks act on *disjoint* primal variables, the data block
    on the fields `u_i` and the regularizer block on the contrast `m`, so with
    the dual metric `Sigma = diag(sigma_dat I, sigma_reg I)` the operator

        K* Sigma K x = ((sigma_dat C*C u_i)_i,  sigma_reg G*G m)

    is block diagonal and the convergence condition `tau ||K* Sigma K|| <= 1`
    is a **max**, not a sum:

        tau * max(sigma_dat ||C||^2, sigma_reg ||G||^2) <= 1.

    Saturating each block separately therefore gives

        sigma_dat = theta / (tau ||C||^2),   sigma_reg = theta / (tau ||G||^2),

    i.e. each dual block is stepped inversely to *its own* norm. The scalar
    rule `sigma = tau = 0.9/||K||` with `||K|| = max(||C||, ||G||)` makes the
    larger block tight and leaves the smaller one slack by exactly the
    imbalance factor, which is the whole quantity of interest here. `tau`
    defaults to `0.9/||K||`, the unpreconditioned primal step, so a paired
    comparison changes only the dual steps.

    (Using the sum form instead would be valid but a factor ~2 conservative,
    and would shrink *both* steps relative to the scalar rule, measuring
    nothing about preconditioning.)

    Returns:
        dict with `tau`, `sigma_dat`, `sigma_reg`, the norms `normC`, `normG`,
        `normK`, and `imbalance = (||G||/||C||)^2`, the quantity that says
        which block dominates, and hence how much preconditioning can buy.

    Note there is no Algorithm 3 counterpart here: `ChambollePock` dualizes
    the PDE and regularizer blocks under a single scalar `sigma`, so giving it
    per-block steps would mean changing the algorithm, not just the operator.
    """
    normC = power_iteration_operator_norm(
        lambda v: pb.C @ v, lambda w: pb.C_star @ w, dim=pb.L, n_iter=n_iter, seed=seed
    )
    G_star = G.conj().T
    normG = power_iteration_operator_norm(
        lambda v: G @ v, lambda w: G_star @ w, dim=pb.P, n_iter=n_iter, seed=seed
    )
    normK = k_operator_norm_algorithm5(pb, G=G, n_iter=n_iter, seed=seed)
    if tau is None:
        tau = 0.9 / normK
    return {
        "tau": tau,
        "sigma_dat": theta / (tau * normC**2),
        "sigma_reg": theta / (tau * normG**2),
        "normC": normC,
        "normG": normG,
        "normK": normK,
        "imbalance": (normG / normC) ** 2,
    }


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


# ---------------------------------------------------------------------------
# Reference solution, conditioning, and equal-work bookkeeping (Sec. 4.8).
#
# Comparing primal-dual schemes against C-NAGD at equal iteration count is
# not a comparison at all: one C-NAGD iteration eliminates the fields, and so
# costs 2I linear solves in A and A^*, while one Algorithm 3 iteration only
# *applies* A. The functions below make the honest comparison possible by
# (a) pinning down the exact minimizer so "converged" means something
# operator-independent, and (b) counting the linear algebra each scheme
# actually performs.
# ---------------------------------------------------------------------------


def reduced_forward_operators(pb: ProblemData):
    """The dense reduced forward operators `Phi_i = C A^-1 B_i` (J x P).

    These are what C-NAGD works with implicitly: eliminating the fields from
    `A u_i = B_i m` turns the constrained problem into the unconstrained
    least-squares problem `min_m sum_i (1/2)||Phi_i m - d_i||^2 + (mu/2)||m||^2`
    on the contrast alone. `P` is a few hundred here, so forming them
    explicitly costs `I` sparse solves with `P` right-hand sides, once.
    """
    A_lu = sp.linalg.splu(pb.A.tocsc())
    return [pb.C @ A_lu.solve(Bi.toarray()) for Bi in pb.B_list]


def exact_regularized_solution(pb: ProblemData, mu, phis=None):
    """Solve the Tikhonov-regularized reduced problem in closed form.

    Returns a dict with the minimizer `m`, the full state `x = [u, m]` it
    lifts to, the optimal value `f_opt` of

        F(m) = sum_i (1/2)||Phi_i m - d_i||^2 + (mu/2)||m||^2,

    and the conditioning `kappa = (||Phi||^2 + mu)/mu` of its Hessian, which
    is the number that sets how many iterations an unaccelerated first-order
    method needs (`O(kappa)`) against an accelerated one (`O(sqrt(kappa))`).

    Every Chambolle-Pock variant in this repository minimizes exactly this
    functional under the constraint, so `m` here is the common target: it is
    what "converged" is measured against, rather than each scheme's own
    fixed-point residual, which is not comparable across schemes.
    """
    phis = reduced_forward_operators(pb) if phis is None else phis
    hessian = sum(phi.conj().T @ phi for phi in phis)
    rhs = sum(phi.conj().T @ di for phi, di in zip(phis, pb.d_list))
    m_opt = np.linalg.solve(hessian + mu * np.eye(pb.P), rhs)
    f_opt = float(
        sum(
            0.5 * np.vdot(phi @ m_opt - di, phi @ m_opt - di).real
            for phi, di in zip(phis, pb.d_list)
        )
        + 0.5 * mu * np.vdot(m_opt, m_opt).real
    )
    A_lu = sp.linalg.splu(pb.A.tocsc())
    u_opt = [A_lu.solve(Bi @ m_opt) for Bi in pb.B_list]
    x_opt = np.concatenate(u_opt + [m_opt])
    phi_norm_sq = float(np.max(np.linalg.eigvalsh(hessian).real))
    return {
        "m": m_opt,
        "x": x_opt,
        "f_opt": f_opt,
        "phi_norm_sq": phi_norm_sq,
        "kappa": (phi_norm_sq + mu) / mu,
    }


def constrained_objective(pb: ProblemData, mu, lambda_tv=None, G=None):
    """`sum_i (1/2)||C u_i - d_i||^2 + (mu/2)||m||^2 [+ lambda ||G m||_1]`,
    evaluated on the full state. This is the functional the Chambolle-Pock
    family minimizes subject to the PDE constraint, so on a feasible iterate
    it is directly comparable to `exact_regularized_solution`'s `f_opt`."""

    def f(x):
        total = 0.0
        for i in range(pb.I):
            r = pb.C @ x[i * pb.L : (i + 1) * pb.L] - pb.d_list[i]
            total += 0.5 * np.vdot(r, r).real
        m = x[pb.I * pb.L : pb.I * pb.L + pb.P]
        total += 0.5 * mu * np.vdot(m, m).real
        if lambda_tv is not None and G is not None:
            total += lambda_tv * float(np.sum(np.abs(G @ m)))
        return total

    return f


def work_counters(algo):
    """Read whatever linear-algebra counters an algorithm exposes.

    `a_solves` counts triangular solves against the cached LU of `A` (or of
    `A^H`), `a_matvecs` applications of the sparse `A`/`A^*` themselves, and
    `c_applies` applications of `C`/`C^*`. A solve and a matvec are not the
    same currency, which is precisely the point: Algorithm 3 spends only
    matvecs, Algorithm 5 spends `4I` solves per iteration in its projection,
    and C-NAGD spends `2I` *unfactored* sparse solves per gradient.
    """
    proj = getattr(algo, "projector", None)
    return {
        "a_solves": getattr(proj, "n_A_solves", 0) + getattr(algo, "n_A_solves", 0),
        "a_matvecs": getattr(proj, "n_A_matvecs", 0) + getattr(algo, "n_A_matvecs", 0),
        "c_applies": getattr(algo, "n_C_applies", 0),
    }


def run_with_tracking(
    algo,
    x0,
    max_iterations,
    P=None,
    objective=None,
    m_ref=None,
    f_opt=None,
    feasibility=None,
    targets=(),
    max_seconds=None,
    stop_threshold=1e-6,
    logger=None,
):
    """Drive `algo.step` directly, recording per-iteration accuracy *and*
    cost, without keeping the iterate history.

    `FixedPointAlgorithm.run` preallocates `(max_iterations+1) x dim`, which
    is 400 MB for the 30000-iteration budgets used here, and it records
    nothing about work done. This runs the same loop and instead records, per
    iteration: the objective, the relative contrast error against `m_ref`,
    the relative primal gap against `f_opt`, the feasibility residual, wall
    time, and the cumulative counters of `work_counters`.

    `targets` is a sequence of relative-error thresholds; for each one the
    returned `hits` dict records the first iteration, wall time and work
    counts at which `||m_k - m_ref|| / ||m_ref||` first fell below it. That
    is the "iterations *and* matvecs to a fixed accuracy" table.

    Returns `(x_final, history_dict, hits_dict)`.
    """
    # C-NAGD iterates on the contrast alone, the primal-dual schemes on the
    # full state, so where `m` sits in `x` has to be told rather than guessed.
    P = getattr(algo, "P", None) if P is None else P
    if P is None:
        raise ValueError("run_with_tracking needs P (the contrast dimension)")
    algo.max_iterations = max_iterations
    algo.store_history = False
    algo.iteration = 0
    x = x0.copy()
    m_ref_norm = np.linalg.norm(m_ref) if m_ref is not None else None

    hist = {k: [] for k in ("iteration", "time", "objective", "rel_m", "rel_gap",
                            "feasibility", "a_solves", "a_matvecs", "c_applies")}
    hits = {}
    remaining = sorted(targets, reverse=True)

    def snapshot(k, t):
        m = x[-P:]
        hist["iteration"].append(k)
        hist["time"].append(t)
        hist["objective"].append(float(objective(x)) if objective else np.nan)
        rel_m = (
            float(np.linalg.norm(m - m_ref) / m_ref_norm)
            if m_ref is not None
            else np.nan
        )
        hist["rel_m"].append(rel_m)
        hist["rel_gap"].append(
            abs(hist["objective"][-1] - f_opt) / abs(f_opt)
            if (objective is not None and f_opt)
            else np.nan
        )
        hist["feasibility"].append(float(feasibility(x)) if feasibility else np.nan)
        for key, value in work_counters(algo).items():
            hist[key].append(value)
        return rel_m

    t0 = time.time()
    snapshot(0, 0.0)
    for k in range(1, max_iterations + 1):
        # `FixedPointAlgorithm.run` calls `is_converged` before every step and
        # the gradient-based algorithms (NAGD, FISTA) compute their gradient
        # inside it, so skipping the call would leave them stepping on a stale
        # gradient. Keep the same order here.
        if algo.is_converged(x, threshold=stop_threshold):
            break
        x = algo.step(x)
        algo.iteration = k
        rel_m = snapshot(k, time.time() - t0)
        while remaining and rel_m is not None and rel_m <= remaining[0]:
            tol = remaining.pop(0)
            hits[tol] = {
                "iteration": k,
                "time": hist["time"][-1],
                **work_counters(algo),
            }
            if logger:
                logger.info(
                    f"{algo.algo_plot_name}: rel_m < {tol:g} at iteration {k} "
                    f"({hits[tol]['a_solves']} A-solves, "
                    f"{hits[tol]['a_matvecs']} A-matvecs)"
                )
        if not np.all(np.isfinite(x)):
            if logger:
                logger.warning(f"{algo.algo_plot_name}: diverged at iteration {k}")
            break
        if max_seconds is not None and hist["time"][-1] > max_seconds:
            if logger:
                logger.info(
                    f"{algo.algo_plot_name}: wall-clock budget of {max_seconds}s "
                    f"reached at iteration {k}"
                )
            break

    algo.iteration = hist["iteration"][-1]
    algo.cv_time = hist["time"][-1]
    algo.f_values = np.asarray(hist["objective"])
    return x, {k: np.asarray(v) for k, v in hist.items()}, hits


def block_step_sizes_algorithm3(pb, G=None, alpha=0.9, n_iter=200, seed=0):
    """Step sizes for `ChambollePock` (Algorithm 3), reported alongside the
    per-block operator norms.

    There is deliberately no block-preconditioned variant here, and the
    reason is the same one that makes partial acceleration impossible for
    Algorithm 3: its two dual blocks are the PDE residual `A u_i - B_i m` and
    the regularizer `G m`, and the first of them touches *both* primal
    variables. `L^* Sigma L` is therefore not block diagonal, the step
    condition does not split into one condition per block, and a `sigma` that
    is generous to the regularizer block is unsafe for the PDE block. This
    returns the scalar `tau = sigma = alpha/||L||` that Algorithm 3 is stuck
    with, plus the norms that show how far from balanced its blocks are.
    """
    normA = power_iteration_operator_norm(
        lambda v: pb.A @ v, lambda w: pb.A_star @ w, dim=pb.L, n_iter=n_iter, seed=seed
    )
    normL = l_operator_norm_algorithm3(pb, G=G, n_iter=n_iter, seed=seed)
    out = {
        "tau": alpha / normL,
        "sigma": alpha / normL,
        "normL": normL,
        "normA": normA,
        "block_diagonal": False,
    }
    if G is not None:
        G_star = G.conj().T
        out["normG"] = power_iteration_operator_norm(
            lambda v: G @ v, lambda w: G_star @ w, dim=pb.P, n_iter=n_iter, seed=seed
        )
        out["imbalance"] = (out["normG"] / normA) ** 2
    return out
