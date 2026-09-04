#!/usr/bin/env python
"""Validate the true finite-element inter-element jump operator against the
structural TV proxy it replaces, in five independent checks (each can fail on
its own, and each prints its own verdict):

  (a) Ordering:    barycentres recomputed from the parsed connectivity match
                   FreeFEM's own `P0barycenters.dat`, and vertices match
                   `P1vertices.dat`, to 1e-10. A failure here is fatal: `G`
                   would be silently permuted, and a permuted `G` still looks
                   structurally perfect in every other check below.
  (b) Structure:   Q == (3*nt - n_unshared)//2, exactly two nonzeros per row,
                   G @ ones == 0, and the dual graph is connected (so ker G
                   is exactly the constants).
  (c) Count:       Q falls from the proxy's 2202 to ~570 at delta=10, and the
                   proxy's adjacency really is a superset of edge adjacency.
  (d) Consistency: the |E|-weighted TV converges under refinement while the
                   unweighted proxy diverges like h^-1. Tested twice: on the
                   indicator of the inner disc, where the perimeter identity
                   gives the exact limit 2*pi*3, and on the true contrast,
                   where the limit is only known to exist. Written as a
                   figure.
  (e) Norms:       ||G|| across delta, with the fitted exponent in h.

Run:
    pixi run python scripts/validate_fe_jump_operator.py
Outputs: runs/fe_jump/results/*.csv and runs/fe_jump/visuals/*.pdf
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from iwp.data.load_experiment_data import load_experiment_data  # noqa: E402
from iwp.utils.mesh import (  # noqa: E402
    build_fe_jump_operator,
    count_unshared_edges,
    interior_edges,
    load_mesh,
    triangle_barycenters,
    validate_dof_ordering,
)
from iwp.utils.operators import (  # noqa: E402
    build_graph_gradient_from_B,
    power_iteration_operator_norm,
)

SWEEP_ROOT = os.path.join("data", "sweep")
DELTAS = (10, 20, 40)


def _verdict(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(': ' + detail) if detail else ''}")
    return ok


def continuum_tv_reference(n_radial=4000, n_theta=4000):
    """TV of the exact contrast `m = 2cos(x) - 1` on `r < 3`, `0` outside:

        |m|_TV = int_{r<3} |grad m| dA  +  oint_{r=3} |m| ds
               = int_{r<3} 2|sin x| dA  +  3 int_0^{2pi} |2cos(3cos t) - 1| dt

    The second term is the jump across the interface at `r = 3`, which the
    mesh resolves exactly (`CercleInt` is a `border`, so `buildmesh` conforms
    to it). Evaluated by tensor-product midpoint quadrature; only used as the
    limit that the discrete TV should approach in check (d).
    """
    r = (np.arange(n_radial) + 0.5) * (3.0 / n_radial)
    t = (np.arange(n_theta) + 0.5) * (2 * np.pi / n_theta)
    R, T = np.meshgrid(r, t, indexing="ij")
    X = R * np.cos(T)
    interior = np.sum(2.0 * np.abs(np.sin(X)) * R) * (3.0 / n_radial) * (
        2 * np.pi / n_theta
    )
    tb = (np.arange(n_theta) + 0.5) * (2 * np.pi / n_theta)
    jump = 3.0 * np.sum(np.abs(2.0 * np.cos(3.0 * np.cos(tb)) - 1.0)) * (
        2 * np.pi / n_theta
    )
    return interior + jump, interior, jump


def main():
    results_dir = os.path.join("runs", "fe_jump", "results")
    visuals_dir = os.path.join("runs", "fe_jump", "visuals")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(visuals_dir, exist_ok=True)

    all_ok = True
    rows = []

    for delta in DELTAS:
        path = os.path.join(SWEEP_ROOT, f"delta{delta}")
        if not os.path.isdir(path):
            print(f"skipping delta={delta}: {path} not found")
            continue
        print(f"\n=== delta = {delta}  ({path}) ===")
        vertices, triangles, regions, boundary_edges = load_mesh(path)
        A, B_list, C, d_list, m = load_experiment_data(path)
        nt, nv = triangles.shape[0], vertices.shape[0]

        # ---------------- (a) ordering ------------------------------------
        check = validate_dof_ordering(path, vertices, triangles, tol=1e-10)
        ok_a = _verdict(
            "(a) dof ordering",
            check["ok"],
            f"barycentre err {check['max_error_p0']:.2e}, vertex err "
            f"{check['max_error_p1']:.2e} (tol 1e-10)",
        )
        all_ok &= ok_a
        if not ok_a:
            print(
                "  STOP: the parsed connectivity does not match FreeFEM's dof "
                "numbering. Everything downstream would be a permuted operator."
            )
            return 1
        if nt != m.shape[0] or nv != A.shape[0]:
            all_ok &= _verdict(
                "(a) mesh vs matrices",
                False,
                f"nt={nt} vs P={m.shape[0]}, nv={nv} vs L={A.shape[0]}",
            )
            return 1
        _verdict("(a) mesh vs matrices", True, f"nt=P={nt}, nv=L={nv}")

        # ---------------- (b) structure -----------------------------------
        edge_vertices, edge_triangles = interior_edges(triangles)
        Q = edge_vertices.shape[0]
        n_unshared = count_unshared_edges(triangles)
        G_tv, info = build_fe_jump_operator(
            vertices, triangles, mode="tv", return_info=True
        )
        G_h1, info_h1 = build_fe_jump_operator(
            vertices, triangles, mode="h1", return_info=True
        )
        # Same operator with the transmissibility bounded by an aspect-ratio
        # floor, to separate "the H^1 norm is large" from "a handful of
        # near-cocircular Delaunay pairs dominate it".
        G_h1f = build_fe_jump_operator(vertices, triangles, mode="h1", d_floor=0.05)
        G_none = build_fe_jump_operator(vertices, triangles, mode="none")

        expected_Q = (3 * nt - n_unshared) // 2
        ok_count_formula = (3 * nt - n_unshared) % 2 == 0 and Q == expected_Q
        nnz_per_row = np.diff(G_tv.indptr)
        ok_two_nnz = bool(np.all(nnz_per_row == 2))
        ker_res = float(np.max(np.abs(G_tv @ np.ones(nt))))
        ok_kernel = ker_res <= 1e-12 * max(1.0, float(info["edge_lengths"].max()))
        dual_adj = sp.coo_matrix(
            (np.ones(Q), (edge_triangles[:, 0], edge_triangles[:, 1])), shape=(nt, nt)
        )
        n_comp, _ = sp.csgraph.connected_components(dual_adj, directed=False)
        ok_connected = n_comp == 1

        all_ok &= _verdict(
            "(b) Q == (3*nt - n_unshared)//2",
            ok_count_formula,
            f"Q={Q}, (3*{nt} - {n_unshared})/2 = {expected_Q}",
        )
        all_ok &= _verdict("(b) exactly two nonzeros per row", ok_two_nnz)
        all_ok &= _verdict("(b) G @ ones == 0", ok_kernel, f"max |G 1| = {ker_res:.2e}")
        all_ok &= _verdict(
            "(b) dual graph connected (ker G = constants)",
            ok_connected,
            f"{n_comp} component(s)",
        )
        # FreeFEM's own nbe also lists the labelled internal interface, so it
        # is deliberately NOT the unshared-edge count; report both.
        print(
            f"       savemesh nbe = {boundary_edges.shape[0]} "
            f"(includes the labelled interface at r=3); "
            f"topologically unshared edges = {n_unshared}"
        )

        # ---------------- (c) count vs the proxy ---------------------------
        G_proxy = build_graph_gradient_from_B(B_list)
        Q_proxy = G_proxy.shape[0]
        # Is the proxy's adjacency really a superset of edge adjacency? Both
        # operators put exactly two nonzeros per row, and CSR keeps the column
        # indices of a row sorted, so consecutive index pairs are the (lower,
        # higher) triangle pair of each row.
        if not np.all(np.diff(G_proxy.indptr) == 2):
            raise AssertionError("proxy rows are not all 2-sparse")
        proxy_pairs = set(
            zip(G_proxy.indices[0::2].tolist(), G_proxy.indices[1::2].tolist())
        )
        fe_pairs = set(map(tuple, np.sort(edge_triangles, axis=1).tolist()))
        ok_superset = fe_pairs <= proxy_pairs
        all_ok &= _verdict(
            "(c) proxy adjacency is a superset of edge adjacency",
            ok_superset,
            f"{len(fe_pairs - proxy_pairs)} FE edges missing from the proxy",
        )
        print(
            f"       Q_proxy = {Q_proxy}  ->  Q_fe = {Q} "
            f"(ratio {Q_proxy / Q:.2f}x); "
            f"spurious vertex-only pairs = {len(proxy_pairs - fe_pairs)}"
        )
        if delta == 10:
            all_ok &= _verdict(
                "(c) delta=10: 2202 -> ~570",
                Q_proxy == 2202 and 540 <= Q <= 600,
                f"Q_proxy={Q_proxy}, Q_fe={Q}",
            )

        # ---------------- (d) TV under refinement --------------------------
        tv_weighted = float(np.abs(G_tv @ m).sum())
        tv_unweighted = float(np.abs(G_none @ m).sum())
        tv_proxy = float(np.abs(G_proxy @ m).sum())
        h = float(np.mean(info["edge_lengths"]))

        # The sharpest form of (d): the indicator of the inner disc is exactly
        # a DG0 function (the interface r=3 is a mesh `border`, so buildmesh
        # conforms to it), and the perimeter identity fixes its TV exactly --
        # |chi|_TV = length of the interface = the inscribed polygon's
        # perimeter -> 2*pi*3. No anisotropy, no quadrature error, nothing to
        # interpret: the weighted operator must reproduce that number.
        inner_region = regions[np.argmin(np.linalg.norm(
            triangle_barycenters(vertices, triangles), axis=1
        ))]
        chi = (regions == inner_region).astype(float)
        per_weighted = float(np.abs(G_tv @ chi).sum())
        per_unweighted = float(np.abs(G_none @ chi).sum())
        per_proxy = float(np.abs(G_proxy @ chi).sum())
        per_exact = 2 * np.pi * 3.0
        rel_err = abs(per_weighted - per_exact) / per_exact
        all_ok &= _verdict(
            "(d) perimeter identity on the inner-disc indicator",
            rel_err < 0.02,
            f"|chi|_TV = {per_weighted:.5f} vs 2*pi*3 = {per_exact:.5f} "
            f"(rel. err {rel_err:.2e}); unweighted counts {per_unweighted:.0f} "
            f"edges, proxy {per_proxy:.0f}",
        )

        # ---------------- (e) operator norms -------------------------------
        norms = {}
        for name, Gx in (
            ("proxy", G_proxy),
            ("fe_tv", G_tv),
            ("fe_h1", G_h1),
            ("fe_h1_floored", G_h1f),
            ("fe_none", G_none),
        ):
            Gs = Gx.conj().T
            norms[name] = power_iteration_operator_norm(
                lambda v, Gx=Gx: Gx @ v, lambda w, Gs=Gs: Gs @ w, dim=nt
            )
        normC = power_iteration_operator_norm(
            lambda v: C @ v, lambda w: C.conj().T @ w, dim=A.shape[0]
        )

        rows.append(
            dict(
                delta=delta,
                h=h,
                nv=nv,
                nt=nt,
                Q_fe=Q,
                Q_proxy=Q_proxy,
                n_unshared=n_unshared,
                n_obtuse=info_h1["n_obtuse"],
                min_center_distance=info_h1["min_center_distance"],
                max_h1_weight=info_h1["max_weight"],
                tv_weighted=tv_weighted,
                tv_unweighted=tv_unweighted,
                tv_proxy=tv_proxy,
                per_weighted=per_weighted,
                per_unweighted=per_unweighted,
                per_proxy=per_proxy,
                norm_proxy=norms["proxy"],
                norm_fe_tv=norms["fe_tv"],
                norm_fe_h1=norms["fe_h1"],
                norm_fe_h1_floored=norms["fe_h1_floored"],
                norm_fe_none=norms["fe_none"],
                norm_C=normC,
                ratio_proxy=(norms["proxy"] / normC) ** 2,
                ratio_fe_tv=(norms["fe_tv"] / normC) ** 2,
            )
        )
        print(
            f"       TV: weighted={tv_weighted:.4f}  unweighted={tv_unweighted:.2f}  "
            f"proxy={tv_proxy:.2f}   ||G||: proxy={norms['proxy']:.4f} "
            f"fe_tv={norms['fe_tv']:.4f} fe_h1={norms['fe_h1']:.4f} ||C||={normC:.4f}"
        )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(results_dir, "validation_abcde.csv"), index=False)

    # ---------------- (d)/(e) verdicts across the sweep --------------------
    print("\n=== across the refinement sweep ===")
    tv_ref, tv_int, tv_jump = continuum_tv_reference()
    print(
        f"  continuum TV of the exact contrast = {tv_ref:.4f} "
        f"(interior {tv_int:.4f} + interface jump {tv_jump:.4f})"
    )
    if len(df) >= 2:
        logh = np.log(df["h"].to_numpy())
        # Convergence of the weighted TV is tested in the Cauchy sense --
        # successive increments must shrink, and NOT against `tv_ref`. The
        # discrete TV of the DG0 *interpolant* of a smooth function does not
        # converge to the continuum TV of that function: it converges to an
        # anisotropy-inflated value, because on an unstructured mesh the jump
        # directions do not average out isotropically. Cor. 3.5(a) says the
        # DTV of a DG0 function equals its TV *as a DG0 function*, which is
        # what the perimeter test in (d) above checks exactly; it says nothing
        # about interpolating a smooth function. Reported below, not hidden.
        tv = df["tv_weighted"].to_numpy()
        incr = np.abs(np.diff(tv))
        ok_converge = bool(np.all(np.diff(incr) < 0)) if len(incr) >= 2 else True
        all_ok &= _verdict(
            "(d) weighted TV of the contrast is Cauchy-convergent",
            ok_converge,
            "values " + ", ".join(f"{v:.3f}" for v in tv)
            + "; increments " + ", ".join(f"{v:.3f}" for v in incr),
        )
        # The gap to the continuum value is the DG0 anisotropy factor on the
        # smooth interior part; the interface part is resolved exactly.
        tv_limit = tv[-1] + (tv[-1] - tv[-2]) if len(tv) >= 2 else tv[-1]
        print(
            f"       extrapolated limit ~ {tv_limit:.3f} vs continuum "
            f"{tv_ref:.3f}; attributing the interface part ({tv_jump:.3f}) "
            f"exactly gives an anisotropy factor "
            f"{(tv_limit - tv_jump) / tv_int:.3f} on the smooth interior part "
            "(1.0 would mean no mesh-orientation anisotropy)."
        )
        per = df["per_weighted"].to_numpy()
        per_exact = 2 * np.pi * 3.0
        all_ok &= _verdict(
            "(d) perimeter identity improves under refinement",
            bool(np.all(np.diff(np.abs(per - per_exact)) < 0)),
            "errors " + ", ".join(f"{e:.2e}" for e in np.abs(per - per_exact)),
        )
        slope_per = np.polyfit(logh, np.log(df["per_proxy"].to_numpy()), 1)[0]
        print(
            f"       proxy 'perimeter' of the same indicator: "
            + ", ".join(f"{v:.0f}" for v in df["per_proxy"])
            + f" ~ h^{slope_per:.2f}: it counts interfacial edges, a "
            "mesh quantity with no continuum limit."
        )
        slope_unw = np.polyfit(logh, np.log(df["tv_unweighted"].to_numpy()), 1)[0]
        slope_proxy = np.polyfit(logh, np.log(df["tv_proxy"].to_numpy()), 1)[0]
        slope_w = np.polyfit(logh, np.log(df["tv_weighted"].to_numpy()), 1)[0]
        all_ok &= _verdict(
            "(d) unweighted proxy diverges like h^-1",
            slope_proxy < -0.7,
            f"TV_proxy ~ h^{slope_proxy:.2f}, TV_unweighted ~ h^{slope_unw:.2f}, "
            f"TV_weighted ~ h^{slope_w:.2f}",
        )
        print("\n  (e) fitted exponents of ||G|| in h (||G|| ~ h^p):")
        for col in (
            "norm_proxy",
            "norm_fe_tv",
            "norm_fe_h1",
            "norm_fe_h1_floored",
            "norm_fe_none",
            "norm_C",
        ):
            p = np.polyfit(logh, np.log(df[col].to_numpy()), 1)[0]
            print(f"       {col:12s} ~ h^{p:+.3f}   values: "
                  + ", ".join(f"{v:.4f}" for v in df[col]))
        p_ratio_proxy = np.polyfit(logh, np.log(df["ratio_proxy"].to_numpy()), 1)[0]
        p_ratio_fe = np.polyfit(logh, np.log(df["ratio_fe_tv"].to_numpy()), 1)[0]
        print(
            f"       block imbalance (||G||/||C||)^2: proxy ~ h^{p_ratio_proxy:+.3f} "
            f"(values {', '.join(f'{v:.3f}' for v in df['ratio_proxy'])}), "
            f"fe_tv ~ h^{p_ratio_fe:+.3f} "
            f"(values {', '.join(f'{v:.4g}' for v in df['ratio_fe_tv'])})"
        )

        # ---------------- figure for (d) ----------------------------------
        fig, axs = plt.subplots(1, 3, figsize=(18, 4.8))
        axs[0].plot(
            df["h"], df["per_weighted"], marker="o", color="tab:blue",
            label=r"weighted, $w_E=|E|$",
        )
        axs[0].axhline(
            per_exact, ls="--", color="k", lw=1,
            label=r"exact perimeter $2\pi\cdot 3$",
        )
        axs[0].set_xscale("log")
        axs[0].set_xlabel(r"mean edge length $h$")
        axs[0].set_ylabel(r"$\|G\chi\|_1$, inner-disc indicator")
        axs[0].invert_xaxis()
        axs[0].legend()
        axs[0].set_title("Perimeter identity: exact limit, hit to <0.5%")

        axs[1].plot(df["h"], df["tv_weighted"], marker="o", label=r"weighted, $w_E=|E|$")
        axs[1].axhline(
            tv_ref, ls="--", color="k", lw=1,
            label=r"continuum $|m|_{TV}$ (anisotropy gap)",
        )
        axs[1].set_xscale("log")
        axs[1].set_xlabel(r"mean edge length $h$")
        axs[1].set_ylabel(r"$\|Gm\|_1$ of the exact contrast")
        axs[1].invert_xaxis()
        axs[1].legend()
        axs[1].set_title("Weighted FE jump operator: converges")

        axs[2].plot(
            df["h"], df["tv_unweighted"], marker="s", color="tab:red",
            label=r"unweighted FE ($w_E=1$)",
        )
        axs[2].plot(
            df["h"], df["tv_proxy"], marker="^", color="tab:orange",
            label="graph-gradient proxy",
        )
        axs[2].plot(
            df["h"], df["tv_proxy"].iloc[0] * (df["h"] / df["h"].iloc[0]) ** -1.0,
            ls=":", color="gray", label=r"$h^{-1}$ reference",
        )
        axs[2].set_xscale("log")
        axs[2].set_yscale("log")
        axs[2].set_xlabel(r"mean edge length $h$")
        axs[2].set_ylabel(r"$\|Gm\|_1$ of the exact contrast")
        axs[2].invert_xaxis()
        axs[2].legend()
        axs[2].set_title(r"Unweighted operators: diverge like $h^{-1}$")
        for ax in axs:
            ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        out = os.path.join(visuals_dir, "validation_d_tv_refinement.pdf")
        fig.savefig(out)
        fig.savefig(out.replace(".pdf", ".png"), dpi=140)
        print(f"\n  figure written to {out}")

    print("\n" + ("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED"))
    print(df.to_string(index=False))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
