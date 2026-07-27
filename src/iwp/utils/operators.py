"""Generic linear-algebra helpers used by the Part 4/5 comparison script and
notebook. Kept separate from `iwp.algorithms.algorithms` because they are
data/geometry utilities (operator-norm estimation, building a TV proxy
operator) rather than algorithm state machines.
"""

import numpy as np
import scipy.sparse as sp

__all__ = [
    "power_iteration_operator_norm",
    "build_graph_gradient_from_B",
    "build_stacked_pde_residual_operators",
]


def power_iteration_operator_norm(matvec, rmatvec, dim, n_iter=200, tol=1e-8, seed=0):
    """Estimate `||K|| = sqrt(largest eigenvalue of K* K)` by power iteration
    on `K* K`, applied purely through the `matvec`/`rmatvec` callables so
    that `K` never has to be assembled densely. This is what the report
    recommends throughout for the operator norms entering the Chambolle-Pock
    step-size conditions (Sec. 3.6, 4.7, 5.2: "we estimate ||L|| by a few
    power iterations on L*L"), and is the only tractable option once the
    stacked dimension I*L grows large enough that dense `eigvals` (used for
    the small I=2 baseline elsewhere in this codebase) becomes prohibitive.

    Args:
        matvec: callable mapping a length-`dim` complex vector to `K x`.
        rmatvec: callable mapping `K x` back to `K* (K x)` (i.e. composed
            with `K*`; equivalently pass the adjoint and this function will
            apply it to `matvec`'s output itself -- see usage in the
            comparison script for both conventions).
        dim: dimension of the input space of `K`.
    Returns:
        float, the estimated spectral norm ``||K||``.
    """
    rng = np.random.default_rng(seed)
    v = (rng.standard_normal(dim) + 1j * rng.standard_normal(dim)).astype(complex)
    v /= np.linalg.norm(v)
    lam_prev = 0.0
    lam = 0.0
    for _ in range(n_iter):
        w = rmatvec(matvec(v))
        norm_w = np.linalg.norm(w)
        if norm_w == 0:
            return 0.0
        v = w / norm_w
        lam = norm_w
        if abs(lam - lam_prev) < tol * max(lam, 1.0):
            break
        lam_prev = lam
    return float(np.sqrt(lam))


def build_graph_gradient_from_B(B_list, dtype=float):
    """Build a *proxy* discrete-gradient (graph-incidence) operator `G` on
    the contrast basis directly from the exported coupling matrices `B_i`,
    since the FreeFEM pipeline (`scripts/GenerateMatrix.edp`) does not export
    mesh connectivity (triangle adjacency / vertex coordinates) needed to
    build the true finite-element inter-element jump operator described in
    Sec. 5.2.

    Two contrast dofs (triangles) `p`, `q` are declared adjacent whenever
    they share support with a common field dof `l` in some `B_i`, i.e.
    whenever `sum_i |B_i|^T |B_i|` has a nonzero off-diagonal entry `(p, q)`.
    This is a superset of true edge-adjacency (it also connects triangles
    that only share a vertex through an intermediate field dof) but requires
    no additional geometric export. `G` is then the signed edge-incidence
    matrix (one row per edge `e = (p, q)`, `G[e, p] = 1`, `G[e, q] = -1`),
    which for the piecewise-constant (P0) contrast basis matches the
    report's remark that TV groups are singleton (`||.||_{2,1} = ||.||_1`,
    Sec. 5.2), so the group-l2,inf-ball dual projection reduces to
    componentwise clipping (`group_size=1` in `group_l2inf_ball_projection`).

    This is a *structural proxy*, not the true FE jump operator: it is only
    used to exercise the Total Variation code path of Algorithms 3-5 in the
    absence of a mesh-connectivity export, and is documented as such
    throughout the comparison notebook.
    """
    P = B_list[0].shape[1]
    adjacency = sp.csr_matrix((P, P))
    for Bi in B_list:
        mask = Bi.tocsr().copy()
        mask.data = np.ones_like(mask.data, dtype=float)
        adjacency = adjacency + (mask.T @ mask)
    adjacency = sp.triu(adjacency, k=1).tocoo()
    n_edges = adjacency.nnz
    if n_edges == 0:
        return sp.csr_matrix((0, P), dtype=dtype)
    rows = np.repeat(np.arange(n_edges), 2)
    cols = np.empty(2 * n_edges, dtype=int)
    cols[0::2] = adjacency.row
    cols[1::2] = adjacency.col
    data = np.tile(np.array([1.0, -1.0], dtype=dtype), n_edges)
    G = sp.coo_matrix((data, (rows, cols)), shape=(n_edges, P)).tocsr()
    return G


def build_stacked_pde_residual_operators(A, B_list):
    """Return `(A_block, B_stacked)` such that the stacked PDE residual
    `r = [A u_0 - B_0 m; ...; A u_{I-1} - B_{I-1} m]` is
    `A_block @ u_stacked - B_stacked @ m`, with `A_block = I_I kron A`
    (block-diagonal) and `B_stacked` the vertical stack of the `B_i`
    (Eq. (47)). Only used by the "spsolve"/"cached_splu" baseline
    projector backends, which need the explicit `I*L x I*L` system
    `E E* = A_block A_block* + B_stacked B_stacked*` (Eq. (48)).
    """
    I = len(B_list)
    A_block = sp.block_diag([A] * I, format="csr")
    B_stacked = sp.vstack(B_list, format="csr")
    return A_block, B_stacked
