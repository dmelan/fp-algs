"""FreeFEM mesh parsing and the *true* finite-element inter-element jump
operator on the piecewise-constant (DG0/P0) contrast space.

This replaces the structural TV proxy of
`iwp.utils.operators.build_graph_gradient_from_B`, which declared two
triangles adjacent whenever they shared *any* P1 field dof, hence also across
a single vertex. In its place we build the operator that the discrete total
variation actually calls for.

Why this is the right object (Herrmann, Herzog, Schmidt, Vidal-Nunez,
Wachsmuth, *Discrete Total Variation with Finite Elements and Applications to
Imaging*, JMIV 61(4):411-431, 2019, DOI 10.1007/s10851-018-0852-7):

* For `r = 0` (piecewise constants) the only basis function on an interior
  edge is `phi_{E,1} = 1`, so the quadrature weight is `c_{E,1} = |E|`
  (Remark 3.3). Equivalently, the perimeter identity: for a union of
  triangles, `|chi_E|_TV = length(E)`.
* Corollary 3.5(a): for `r = 0` the discretization is *exact*,
  `|u|_TV = |u|_DTV` for every `u` in DG0, since the jump is constant along
  each edge. The regularizer carries no discretization error at all once the
  weights are right.
* The dual constraints for `r = 0` are the scalar bounds
  `int_E |p . n_E| dS <= |E| |n_E|_s` (Eq. (1.4)): the singleton-group case,
  so `group_l2inf_ball_projection(..., group_size=1)` stays correct and only
  the ball radius picks up the `|E|` factor. Nothing in Algorithms 3/4/5
  changes: only the matrix and the radius do.

We use `s = 2` (isotropic): then `|[[u]]|_s = |[[u]]| |n_E|_2 = |[[u]]|`, so
the weight is exactly `|E|`. Picking `s = 1` would make the weight
orientation-dependent and the regularizer anisotropic.

Every hypothesis of the reference is met literally by this pipeline:
`scripts/GenerateMatrix.edp` declares `fespace FS0(Th, P0)` and
`FS0<complex> m = n1 - n0`, so the contrast basis *is* DG0; FreeFEM's
`buildmesh` produces a geometrically conforming triangulation with no hanging
nodes; and the regularizer acts on `m` alone, so TV is only ever needed on
the P0 space.

For the *first-order Tikhonov control* a different weight is required,
`w_E = sqrt(|E| / d_E)` with `d_E` the distance between the two cell centres:
this is the two-point flux transmissibility of the classical cell-centred
finite volume method (Eymard, Gallouet, Herbin, *Finite Volume Methods*,
Handbook of Numerical Analysis VII, North-Holland, 2000, pp. 713-1020), whose
quadratic form `sum_E (|E|/d_E) |m_K - m_L|^2` is the standard discrete H^1
seminorm. Using `|E|` for both would make the "H^1" control a length-weighted
sum of squared jumps, which is not an H^1 seminorm and would invalidate the
comparison that control exists to make.
"""

import logging
import os

import numpy as np
import scipy.sparse as sp

logger = logging.getLogger("iwp")

__all__ = [
    "read_freefem_mesh",
    "read_freefem_coordinate_pairs",
    "interior_edges",
    "count_unshared_edges",
    "triangle_barycenters",
    "triangle_circumcenters",
    "edge_lengths",
    "build_fe_jump_operator",
    "load_mesh",
    "load_dims",
    "load_contrast_mesh",
    "validate_dof_ordering",
    "make_triangulation",
]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def read_freefem_mesh(path, return_labels=False):
    """Parse a 2D mesh written by FreeFEM's `savemesh`.

    Format (all indices 1-based in the file, 0-based in what we return)::

        nv nt nbe
        x y label            x nv
        v1 v2 v3 region      x nt
        v1 v2 label          x nbe

    Note that FreeFEM's `nbe` counts *labelled* boundary elements, which for
    `buildmesh(CercleExt(...) + CercleInt(...))` includes the internal
    interface `CercleInt` (label 2) even though those edges are shared by two
    triangles. So `nbe` is **not** the number of topologically unshared
    edges. Use `count_unshared_edges` for that (see `interior_edges`).

    Args:
        path: path to the `.msh` file.
        return_labels: if True, also return `(vertex_labels, boundary_labels)`.

    Returns:
        `(vertices (nv, 2) float, triangles (nt, 3) int 0-based,
          regions (nt,) int, boundary_edges (nbe, 2) int 0-based)`,
        plus `(vertex_labels, boundary_labels)` if `return_labels`.
    """
    with open(path, "r") as f:
        tokens = f.read().split()
    if len(tokens) < 3:
        raise ValueError(f"{path}: file too short to contain a mesh header")

    nv, nt, nbe = (int(tokens[0]), int(tokens[1]), int(tokens[2]))
    expected = 3 + 3 * nv + 4 * nt + 3 * nbe
    if len(tokens) != expected:
        raise ValueError(
            f"{path}: token count {len(tokens)} does not match header "
            f"nv={nv} nt={nt} nbe={nbe} (expected {expected}). "
            "The file is not a 2D FreeFEM `savemesh` output, or is truncated."
        )

    body = np.asarray(tokens[3:], dtype=object)
    off = 0
    vblock = np.array(body[off : off + 3 * nv], dtype=float).reshape(nv, 3)
    off += 3 * nv
    tblock = np.array(body[off : off + 4 * nt], dtype=float).reshape(nt, 4)
    off += 4 * nt
    bblock = np.array(body[off : off + 3 * nbe], dtype=float).reshape(nbe, 3)

    vertices = vblock[:, :2].astype(float)
    vertex_labels = vblock[:, 2].astype(int)
    triangles = tblock[:, :3].astype(int) - 1
    regions = tblock[:, 3].astype(int)
    boundary_edges = bblock[:, :2].astype(int) - 1
    boundary_labels = bblock[:, 2].astype(int)

    if triangles.min() < 0 or triangles.max() >= nv:
        raise ValueError(f"{path}: triangle vertex index out of range [1, {nv}]")
    if nbe and (boundary_edges.min() < 0 or boundary_edges.max() >= nv):
        raise ValueError(f"{path}: boundary edge vertex index out of range [1, {nv}]")

    if return_labels:
        return vertices, triangles, regions, boundary_edges, vertex_labels, (
            boundary_labels
        )
    return vertices, triangles, regions, boundary_edges


def read_freefem_coordinate_pairs(path, n_expected=None):
    """Read the `f << ax[] << ay[]` coordinate dumps written by the `.edp`
    scripts (`P0barycenters.dat`, `P1vertices.dat`): two consecutive FreeFEM
    `real[int]` blocks, each a length header followed by that many values.

    Returns an `(n, 2)` array of `(x, y)`.
    """
    with open(path, "r") as f:
        tokens = f.read().split()
    n = int(tokens[0])
    if n_expected is not None and n != n_expected:
        raise ValueError(f"{path}: expected length {n_expected}, file declares {n}")
    xs = np.array(tokens[1 : 1 + n], dtype=float)
    rest = tokens[1 + n :]
    if not rest:
        raise ValueError(f"{path}: only one coordinate array found, expected two")
    n2 = int(rest[0])
    if n2 != n:
        raise ValueError(f"{path}: coordinate arrays have different lengths {n}, {n2}")
    ys = np.array(rest[1 : 1 + n], dtype=float)
    if len(rest) != 1 + n:
        raise ValueError(f"{path}: trailing tokens after the second coordinate array")
    return np.column_stack([xs, ys])


# ---------------------------------------------------------------------------
# Connectivity
# ---------------------------------------------------------------------------


def _sorted_edge_table(triangles):
    """`(edges_sorted (3nt, 2), triangle_of_edge (3nt,))` for the three edges
    of every triangle, each vertex pair sorted ascending."""
    triangles = np.asarray(triangles)
    nt = triangles.shape[0]
    edges = np.concatenate(
        [triangles[:, [1, 2]], triangles[:, [2, 0]], triangles[:, [0, 1]]], axis=0
    )
    triangle_of_edge = np.tile(np.arange(nt), 3)
    return np.sort(edges, axis=1), triangle_of_edge


def interior_edges(triangles):
    """Interior (two-triangle) edges of a conforming triangulation.

    Returns:
        `(edge_vertices (Q, 2), edge_triangles (Q, 2))`, both 0-based.
        `edge_vertices` rows are sorted vertex pairs, in lexicographic order,
        and `edge_triangles` is ordered `(lower index, higher index)` so that
        the jump sign convention `[[m]]_E = m_{T-} - m_{T+}` is fixed and
        reproducible run to run.

    Raises:
        ValueError: if any edge is shared by more than two triangles (the mesh
            is then not a valid conforming 2-manifold triangulation).
    """
    edges_sorted, triangle_of_edge = _sorted_edge_table(triangles)
    uniq, inverse, counts = np.unique(
        edges_sorted, axis=0, return_inverse=True, return_counts=True
    )
    inverse = np.asarray(inverse).ravel()
    if counts.size and counts.max() > 2:
        bad = uniq[counts > 2]
        raise ValueError(
            f"{bad.shape[0]} edge(s) shared by more than two triangles, e.g. "
            f"vertices {bad[0].tolist()}: mesh is not a conforming triangulation."
        )

    order = np.argsort(inverse, kind="stable")
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    interior = np.flatnonzero(counts == 2)
    t0 = triangle_of_edge[order[starts[interior]]]
    t1 = triangle_of_edge[order[starts[interior] + 1]]
    edge_triangles = np.column_stack([np.minimum(t0, t1), np.maximum(t0, t1)])
    return uniq[interior], edge_triangles


def count_unshared_edges(triangles):
    """Number of edges belonging to exactly one triangle, i.e. the *true*
    topological boundary of the triangulated domain. Distinct from the `nbe`
    of `savemesh`, which also lists labelled internal interfaces."""
    edges_sorted, _ = _sorted_edge_table(triangles)
    _, counts = np.unique(edges_sorted, axis=0, return_counts=True)
    return int(np.sum(counts == 1))


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def triangle_barycenters(vertices, triangles):
    """`(nt, 2)` element barycentres, the points a FreeFEM P0 interpolation
    samples at and hence what validation (a) compares against."""
    return np.asarray(vertices)[np.asarray(triangles)].mean(axis=1)


def triangle_circumcenters(vertices, triangles):
    """`(nt, 2)` element circumcentres. These, not the barycentres, are the
    admissible TPFA cell centres for a Delaunay mesh: the segment joining the
    circumcentres of two triangles sharing an edge is orthogonal to that edge,
    which is exactly the consistency condition of the two-point flux
    approximation (Eymard-Gallouet-Herbin). For an obtuse triangle the
    circumcentre falls outside the element; see `build_fe_jump_operator`,
    which detects and reports those."""
    v = np.asarray(vertices, dtype=float)[np.asarray(triangles)]
    ax, ay = v[:, 0, 0], v[:, 0, 1]
    bx, by = v[:, 1, 0], v[:, 1, 1]
    cx, cy = v[:, 2, 0], v[:, 2, 1]
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    a2, b2, c2 = ax**2 + ay**2, bx**2 + by**2, cx**2 + cy**2
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    return np.column_stack([ux, uy])


def _is_obtuse(vertices, triangles):
    """Boolean mask of triangles with an obtuse angle (circumcentre outside)."""
    v = np.asarray(vertices, dtype=float)[np.asarray(triangles)]
    out = np.zeros(v.shape[0], dtype=bool)
    for i in range(3):
        e1 = v[:, (i + 1) % 3] - v[:, i]
        e2 = v[:, (i + 2) % 3] - v[:, i]
        out |= np.sum(e1 * e2, axis=1) < 0.0
    return out


def edge_lengths(vertices, edge_vertices):
    """`|E|` for each edge, i.e. the TV quadrature weight `c_{E,1}` for r=0."""
    v = np.asarray(vertices, dtype=float)
    seg = v[np.asarray(edge_vertices)[:, 1]] - v[np.asarray(edge_vertices)[:, 0]]
    return np.linalg.norm(seg, axis=1)


# ---------------------------------------------------------------------------
# The operator
# ---------------------------------------------------------------------------


def build_fe_jump_operator(
    vertices,
    triangles,
    mode="tv",
    center="circumcenter",
    d_floor=0.0,
    return_info=False,
    dtype=float,
):
    """Signed inter-element jump operator `G = W @ G0` on the P0 (DG0)
    contrast space, `G0` the unweighted incidence with one row per interior
    edge (`G0[e, T-] = +1`, `G0[e, T+] = -1`, `T- < T+`) and `W = diag(w_E)`.

    ============  ==========================  ===============================
    mode          weight `w_E`                use
    ============  ==========================  ===============================
    ``"tv"``      `|E|`                       discrete TV, `s = 2` (exact for
                                              DG0 by Cor. 3.5(a))
    ``"h1"``      `sqrt(|E| / d_E)`           discrete H^1 seminorm (TPFA)
    ``"none"``    `1`                         diagnostics only; the geometry-
                                              free operator, comparable to
                                              the old proxy's `+-1` entries
    ============  ==========================  ===============================

    `d_E` is the distance between the two cell centres. For `"h1"` the
    centres default to the **circumcentres** (TPFA admissibility); pass
    `center="barycenter"` for the non-admissible variant, which is what
    validation (a) uses as a cross-check.

    Obtuse triangles are handled explicitly. Their circumcentre lies outside
    the element, and for a *nearly cocircular* pair of triangles the two
    circumcentres almost coincide, so `d_E -> 0` and the transmissibility
    `|E|/d_E` blows up: on this geometry `min d_E` is 7.7e-3 at delta=10 but
    4.2e-9 at delta=40, and a handful of such edges alone raise `||G_h1||`
    from ~15 to ~1e4. That is real geometry (the Delaunay triangulation is
    non-unique at a cocircular quadrilateral), not a parsing bug, so it is
    reported rather than hidden, through `info["min_center_distance"]`,
    `info["max_weight"]` and `info["n_below_floor"]`, and `d_floor` lets a
    caller bound it when the operator norm has to feed a step size.

    Args:
        d_floor: if > 0, floor `d_E` at `d_floor * |E|`, capping the
            transmissibility at `1/d_floor` and hence `w_E` at
            `sqrt(1/d_floor)`. Scale-free (a pure aspect-ratio bound) so it
            behaves consistently under refinement. The default `0.0` applies
            no flooring, i.e. the plain TPFA construction.
        return_info: if True, return `(G, info)` with `info` carrying the
            edge arrays, weights and the obtuse/degeneracy diagnostics.

    Returns:
        `scipy.sparse.csr_matrix` of shape `(Q, P)`, `Q` = number of interior
        edges, `P` = number of triangles.
    """
    vertices = np.asarray(vertices, dtype=float)
    triangles = np.asarray(triangles)
    P = triangles.shape[0]
    edge_vertices, edge_triangles = interior_edges(triangles)
    Q = edge_vertices.shape[0]

    lengths = edge_lengths(vertices, edge_vertices)
    info = {
        "edge_vertices": edge_vertices,
        "edge_triangles": edge_triangles,
        "edge_lengths": lengths,
        "n_obtuse": 0,
        "n_below_floor": 0,
        "mode": mode,
    }

    if mode == "none":
        weights = np.ones(Q)
    elif mode == "tv":
        weights = lengths
    elif mode == "h1":
        bary = triangle_barycenters(vertices, triangles)
        if center == "circumcenter":
            centers = triangle_circumcenters(vertices, triangles)
        elif center == "barycenter":
            centers = bary
        else:
            raise ValueError(f"Unknown center: {center!r}")
        d = np.linalg.norm(
            centers[edge_triangles[:, 1]] - centers[edge_triangles[:, 0]], axis=1
        )
        info["min_center_distance"] = float(d.min()) if Q else float("nan")
        obtuse = _is_obtuse(vertices, triangles)
        info["n_obtuse"] = int(obtuse.sum())
        n_below = 0
        if d_floor > 0.0:
            floor = d_floor * lengths
            n_below = int(np.sum(d < floor))
            d = np.maximum(d, floor)
        elif np.any(d <= 0.0):
            raise ValueError(
                "Degenerate (zero) cell-centre distance: the triangulation has "
                "exactly cocircular neighbours. Pass d_floor > 0 to bound the "
                "transmissibility."
            )
        info["n_below_floor"] = n_below
        info["center_distances"] = d
        weights = np.sqrt(lengths / d)
        info["max_weight"] = float(weights.max()) if Q else float("nan")
        if n_below:
            logger.info(
                f"build_fe_jump_operator(mode='h1'): floored d_E on {n_below}/{Q} "
                f"near-cocircular edges (d_floor={d_floor}), capping w_E at "
                f"{np.sqrt(1.0 / d_floor):.3g}."
            )
        elif Q and weights.max() > 50.0:
            logger.warning(
                f"build_fe_jump_operator(mode='h1'): max w_E = {weights.max():.3g} "
                f"(min d_E = {info['min_center_distance']:.2e}) from near-cocircular "
                "triangle pairs. ||G_h1|| is dominated by those few edges; pass "
                "d_floor (e.g. 0.05) if this norm feeds a step size."
            )
    else:
        raise ValueError(f"Unknown mode: {mode!r} (expected 'tv', 'h1' or 'none')")

    info["weights"] = weights
    rows = np.repeat(np.arange(Q), 2)
    cols = edge_triangles.reshape(-1)
    data = np.empty(2 * Q, dtype=dtype)
    data[0::2] = weights
    data[1::2] = -weights
    G = sp.coo_matrix((data, (rows, cols)), shape=(Q, P)).tocsr()
    return (G, info) if return_info else G


# ---------------------------------------------------------------------------
# Dataset-level helpers
# ---------------------------------------------------------------------------


def load_mesh(data_path, filename="mesh.msh"):
    """Read `<data_path>/mesh.msh`, raising a message that says how to
    produce it if the dataset predates the connectivity export."""
    path = os.path.join(data_path, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No mesh export at {path}. Regenerate the dataset with the current "
            "scripts/GenerateMatrix.edp or scripts/GenerateMatrixSweep.edp, which "
            "call savemesh(Th, OutputDir + 'mesh.msh')."
        )
    return read_freefem_mesh(path)


def load_dims(data_path, filename="dims.txt"):
    """Read the `dims.txt` written by `scripts/GenerateMatrixSweep.edp`.

    A dataset generated with a decoupled contrast mesh is no longer
    self-describing from its matrix shapes alone (`delta` and `delta_m` are
    both needed to say which regime it is in), so the generator writes them
    out. Returns None for the older datasets that predate the file, which
    callers should read as "single mesh, delta unknown".
    """
    path = os.path.join(data_path, filename)
    if not os.path.exists(path):
        return None
    out = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                out[parts[0]] = int(parts[1])
    return out


def load_contrast_mesh(data_path):
    """Read the mesh the P0 contrast lives on.

    Returns `(vertices, triangles, regions, boundary_edges, decoupled)`. When
    the dataset carries a `mesh_contrast.msh` distinct from `mesh.msh` (i.e.
    it was generated with `-delta_m` different from `-delta`), that is the
    mesh returned and `decoupled` is True; otherwise the field mesh is
    returned and `decoupled` is False. Every consumer of the contrast
    connectivity -- the jump operator `G`, contrast-space plotting, the
    `P == n_triangles` consistency check -- must go through this rather than
    through `load_mesh`, which returns the *field* mesh.
    """
    field = load_mesh(data_path)
    path = os.path.join(data_path, "mesh_contrast.msh")
    if not os.path.exists(path):
        return (*field, False)
    contrast = read_freefem_mesh(path)
    decoupled = contrast[1].shape[0] != field[1].shape[0] or not np.array_equal(
        contrast[0], field[0]
    )
    return (*contrast, decoupled)


def validate_dof_ordering(
    data_path,
    vertices,
    triangles,
    tol=1e-10,
    contrast_vertices=None,
    contrast_triangles=None,
):
    """Validation (a): check that the parsed connectivity reproduces FreeFEM's
    own dof ordering.

    A P0 interpolation of `x`/`y` samples at the element barycentre and a P1
    one at the vertex, so `P0barycenters.dat` and `P1vertices.dat` pin the
    dof <-> element and dof <-> vertex maps exactly. If this fails, `G` would
    be a silently permuted operator, structurally plausible but wrong, so
    callers should treat a failure as fatal.

    `vertices`/`triangles` are the *field* mesh, against which `P1vertices.dat`
    is checked. `contrast_vertices`/`contrast_triangles` are the mesh the P0
    contrast lives on, against which `P0barycenters.dat` is checked; they
    default to the field mesh, which is the single-mesh case. On a dataset
    generated with `-delta_m`, passing the field mesh for both would compare
    arrays of different lengths and fail loudly rather than silently, but
    passing the right one is the caller's job.

    Returns:
        dict with `max_error_p0`, `max_error_p1`, `ok`.
    """
    out = {"max_error_p0": None, "max_error_p1": None, "ok": True}
    if contrast_vertices is None:
        contrast_vertices = vertices
    if contrast_triangles is None:
        contrast_triangles = triangles

    p0_file = os.path.join(data_path, "P0barycenters.dat")
    if os.path.exists(p0_file):
        ref = read_freefem_coordinate_pairs(
            p0_file, n_expected=contrast_triangles.shape[0]
        )
        err = float(
            np.max(
                np.abs(
                    triangle_barycenters(contrast_vertices, contrast_triangles) - ref
                )
            )
        )
        out["max_error_p0"] = err
        out["ok"] &= err <= tol
    p1_file = os.path.join(data_path, "P1vertices.dat")
    if os.path.exists(p1_file):
        ref = read_freefem_coordinate_pairs(p1_file, n_expected=vertices.shape[0])
        err = float(np.max(np.abs(np.asarray(vertices) - ref)))
        out["max_error_p1"] = err
        out["ok"] &= err <= tol
    if out["max_error_p0"] is None and out["max_error_p1"] is None:
        raise FileNotFoundError(
            f"Neither P0barycenters.dat nor P1vertices.dat found in {data_path}; "
            "cannot validate the dof ordering. Regenerate the dataset."
        )
    return out


def make_triangulation(vertices, triangles):
    """`matplotlib.tri.Triangulation` for the mesh, so P0 fields can be drawn
    on the actual annulus with `tripcolor(tri, facecolors=values)` instead of
    being plotted against a contrast-dof index."""
    from matplotlib.tri import Triangulation

    vertices = np.asarray(vertices, dtype=float)
    return Triangulation(vertices[:, 0], vertices[:, 1], np.asarray(triangles))
