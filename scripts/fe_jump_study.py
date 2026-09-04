#!/usr/bin/env python
"""Paired proxy-vs-true-operator re-runs of the experiments that depend on the
regularizer operator `G`.

Every experiment is run twice on identical data with identical seeds, once
with `G_mode="proxy"` (the historical graph-gradient proxy built from the
`B_i` sparsity pattern) and once with `G_mode="fe_tv"` (the edge-length
weighted finite-element inter-element jump operator), so the two columns of
every table differ *only* by the operator:

  block:    Sec. 5.3, block metric: ||G||, ||C||, the imbalance (||G||/||C||)^2
             and what block preconditioning (Eq. (56)) actually buys.
  reg:      Sec. 6.1, regularizer sweep. The grid is re-swept, not
             transplanted: the weighting changes ||G m||_1 by O(h), so the
             optimum moves by about that factor, and the previous optimum
             1e-3 sat at the edge of its grid. Extended downward to 1e-7.
  mesh:     Sec. 6.2, mesh sweep at delta = 10, 20 (add 40 with --deltas
             10,20,40; it costs tens of minutes per arm on a small machine).
  transfer: Sec. 6.3, whether the tuned lambda survives a change of mesh
             (it should for the |E|-weighted operator, and must not for the
             proxy, whose ||G m||_1 diverges like h^-1).
  noise:    Sec. 9, noise robustness.

Run:
    pixi run python scripts/fe_jump_study.py [--only block,reg,mesh,noise]
                                             [--quick]
Outputs land in runs/fe_jump/{results,visuals}.
"""

import argparse
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from iwp.algorithms.algorithms import (  # noqa: E402
    AffineConstraintProjector,
    ChambollePock,
    NesterovAcceleratedGradientDescent,
    ProjectedChambollePock,
    make_tikhonov_dual_prox,
    make_tv_dual_prox,
)
from iwp.experiments.comparison import (  # noqa: E402
    ProblemData,
    block_step_sizes_algorithm5,
    build_regularizer_operator,
    get_dJ_3,
    get_J_3,
    get_K_J_3,
    k_operator_norm_algorithm5,
    l_operator_norm_algorithm3,
    load_problem,
)
from iwp.utils.logger import setup_logger  # noqa: E402
from iwp.utils.operators import power_iteration_operator_norm  # noqa: E402

SWEEP_ROOT = os.path.join("data", "sweep")
RESULTS = os.path.join("runs", "fe_jump", "results")
VISUALS = os.path.join("runs", "fe_jump", "visuals")
MODES = ("proxy", "fe_tv")
MODE_LABEL = {"proxy": "graph-gradient proxy", "fe_tv": r"FE jump, $w_E=|E|$"}


def data_fidelity(pb, d_list=None):
    """`sum_i (1/2)||C u_i - d_i||^2`, the term all of Algorithms 3/4/5
    actually minimize; the regularizer is reported separately."""
    d_list = pb.d_list if d_list is None else d_list

    def f(x):
        return sum(
            0.5 * np.vdot(r, r).real
            for r in (
                pb.C @ x[i * pb.L : (i + 1) * pb.L] - d_list[i] for i in range(pb.I)
            )
        )

    return f


def mse_mae(x, m_true, P):
    m_pred = x[-P:] if x.shape[0] != P else x
    return (
        float(np.mean(np.abs(m_pred - m_true) ** 2)),
        float(np.mean(np.abs(m_pred - m_true))),
    )


def make_projector(pb, max_dense_P=2000):
    """Exact SMW where the dense `P x P` capacitance is small, and the
    matrix-free CG variant beyond that.

    The threshold is about speed, not only memory. Measured at delta=40
    (L=3461, P=6720) on a 372 GB machine, where the 722 MB capacitance fits
    easily: 100 iterations cost 21.0s with "smw" against 22.2s with "smw_cg",
    while setup costs 62.4s against 0.7s, and both reach the same iterate
    (MSE 0.693926 either way). The dense capacitance solve is memory-bandwidth
    bound at that size, so it buys back none of its setup. Raising this
    threshold is therefore not worth it even with the RAM to spare."""
    method = "smw" if pb.P <= max_dense_P else "smw_cg"
    kwargs = {} if method == "smw" else {"cg_gamma": 0.7}
    return AffineConstraintProjector(pb.A, pb.B_list, method=method, **kwargs)


def run_alg5(pb, G, lambda_tv, tau, sigma_dat, sigma_reg, iters, d_list=None,
             projector=None, name="Alg5"):
    d_list = pb.d_list if d_list is None else d_list
    projector = projector or make_projector(pb)
    algo = ProjectedChambollePock(
        exp_name="fe_jump", algo_plot_name=name, f=data_fidelity(pb, d_list),
        C=pb.C, d=d_list, I=pb.I, L=pb.L, P=pb.P,
        tau=tau, sigma_dat=sigma_dat, sigma_reg=sigma_reg, projector=projector,
        reg_mode="tv", G=G, lambda_tv=lambda_tv,
    )
    t0 = time.time()
    x = algo.run(
        x0=np.zeros(pb.I * pb.L + pb.P, dtype=complex),
        max_iterations=iters,
        store_history=False,
    )
    return x, algo, time.time() - t0


def run_alg3(pb, G, lambda_tv, tau, sigma, iters, d_list=None, name="Alg3"):
    d_list = pb.d_list if d_list is None else d_list
    algo = ChambollePock(
        exp_name="fe_jump", algo_plot_name=name, f=data_fidelity(pb, d_list),
        A=pb.A, B=pb.B_list, C=pb.C, G=G, d=d_list, I=pb.I, L=pb.L, P=pb.P,
        tau=tau, sigma=sigma, prox_dual_reg=make_tv_dual_prox(lambda_tv),
    )
    t0 = time.time()
    x = algo.run(
        x0=np.zeros(pb.I * pb.L + pb.P, dtype=complex),
        max_iterations=iters,
        store_history=False,
    )
    return x, algo, time.time() - t0


# ===========================================================================
# Sec. 5.3: block metric
# ===========================================================================


def study_block(logger, deltas=(10, 20), lambda_tv=1e-3, iters=3000):
    """Which dual block dominates, and does block preconditioning matter more
    with the true operator? Runs Algorithm 5 with a single scalar sigma
    (Eq. (33)) and with per-block sigmas (Eq. (56)), for both operators."""
    logger.info("=== Sec. 5.3: block metric, proxy vs FE jump operator ===")
    rows = []
    for delta in deltas:
        path = os.path.join(SWEEP_ROOT, f"delta{delta}")
        if not os.path.isdir(path):
            continue
        for mode in MODES:
            pb = load_problem(path, G_mode=mode)
            G = pb.G
            bs = block_step_sizes_algorithm5(pb, G)
            tau = bs["tau"]
            projector = make_projector(pb)
            for scheme, (s_dat, s_reg) in (
                ("scalar", (0.9 / bs["normK"], 0.9 / bs["normK"])),
                ("block", (bs["sigma_dat"], bs["sigma_reg"])),
            ):
                x, algo, wall = run_alg5(
                    pb, G, lambda_tv,
                    tau if scheme == "block" else 0.9 / bs["normK"],
                    s_dat, s_reg, iters, projector=projector,
                    name=f"Alg5-{mode}-{scheme}-d{delta}",
                )
                mse, mae = mse_mae(x, pb.m, pb.P)
                rows.append(dict(
                    delta=delta, G_mode=mode, scheme=scheme, Q=G.shape[0],
                    normG=bs["normG"], normC=bs["normC"], normK=bs["normK"],
                    imbalance=bs["imbalance"], tau=tau,
                    sigma_dat=s_dat, sigma_reg=s_reg,
                    objective=float(algo.f_values[algo.iteration]),
                    mse=mse, mae=mae, wall_time=wall,
                ))
                logger.info(
                    f"  delta={delta} {mode:6s} {scheme:6s}: ||G||={bs['normG']:.4f} "
                    f"||C||={bs['normC']:.4f} imbalance={bs['imbalance']:.4g} "
                    f"obj={rows[-1]['objective']:.5g} mse={mse:.5g}"
                )
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "sec53_block_metric.csv"), index=False)

    piv = df.pivot_table(
        index=["delta", "G_mode"], columns="scheme", values=["objective", "mse"]
    )
    piv["gain_objective"] = piv[("objective", "scalar")] / piv[("objective", "block")]
    # Flatten the MultiIndex columns before writing: a two-row CSV header is
    # unreadable both to pandas' own reader and to a human.
    flat = piv.copy()
    flat.columns = [
        "_".join(c).strip("_") if isinstance(c, tuple) else c for c in flat.columns
    ]
    flat.reset_index().to_csv(
        os.path.join(RESULTS, "sec53_block_metric_pivot.csv"), index=False
    )
    logger.info("Block-preconditioning gain (scalar/block objective):\n" + piv.to_string())

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    for mode in MODES:
        sub = df[(df.G_mode == mode) & (df.scheme == "scalar")]
        axs[0].plot(sub["delta"], sub["imbalance"], marker="o", label=MODE_LABEL[mode])
    axs[0].axhline(1.0, ls="--", color="k", lw=1, label="balanced blocks")
    axs[0].set_yscale("log")
    axs[0].set_xlabel(r"mesh density $\delta$")
    axs[0].set_ylabel(r"$(\|G\|/\|C\|)^2$")
    axs[0].set_title("Which dual block dominates")
    axs[0].legend()
    for mode in MODES:
        sub = piv.xs(mode, level="G_mode")
        axs[1].plot(
            sub.index.get_level_values("delta"), sub["gain_objective"],
            marker="o", label=MODE_LABEL[mode],
        )
    axs[1].axhline(1.0, ls="--", color="k", lw=1, label="no gain")
    axs[1].set_xlabel(r"mesh density $\delta$")
    axs[1].set_ylabel("objective(scalar) / objective(block)")
    axs[1].set_title(f"What block preconditioning buys ({iters} iters)")
    axs[1].legend()
    for ax in axs:
        ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(VISUALS, "sec53_block_metric.pdf"))
    fig.savefig(os.path.join(VISUALS, "sec53_block_metric.png"), dpi=140)
    plt.close(fig)
    return df


# ===========================================================================
# Sec. 6.1: regularizer sweep, re-swept
# ===========================================================================


def study_regularizer(logger, data_path="data", iters=10000,
                      lambda_grid=(1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2)):
    """Sweep lambda_TV for both operators and both algorithms.

    The grid must *bracket* the optimum for each operator separately, and the
    two optima are not in the same place: `||G m||_1` is ~10x smaller with the
    |E|-weighted operator (77.5 vs 765.8 at delta=10), so the same nominal
    lambda applies a ~10x weaker penalty and the FE optimum sits about a
    decade *higher* than the proxy's. Re-running with a different
    `lambda_grid` merges into the existing CSV rather than replacing it, so
    the grid can be extended in either direction until both optima are
    interior.
    """
    logger.info("=== Sec. 6.1: lambda_TV sweep, proxy vs FE jump operator ===")
    rows = []
    for mode in MODES:
        pb = load_problem(data_path, G_mode=mode)
        G = pb.G
        l3 = l_operator_norm_algorithm3(pb, G=G)
        k5 = k_operator_norm_algorithm5(pb, G=G)
        projector = make_projector(pb)
        for lam in lambda_grid:
            x3, a3, w3 = run_alg3(pb, G, lam, 0.9 / l3, 0.9 / l3, iters,
                                  name=f"Alg3-{mode}-{lam:.0e}")
            mse3, mae3 = mse_mae(x3, pb.m, pb.P)
            x5, a5, w5 = run_alg5(pb, G, lam, 0.9 / k5, 0.9 / k5, 0.9 / k5, iters,
                                  projector=projector, name=f"Alg5-{mode}-{lam:.0e}")
            mse5, mae5 = mse_mae(x5, pb.m, pb.P)
            for algo_name, mse, mae, wall, tv_term in (
                ("Alg3", mse3, mae3, w3, float(np.abs(G @ x3[-pb.P:]).sum())),
                ("Alg5", mse5, mae5, w5, float(np.abs(G @ x5[-pb.P:]).sum())),
            ):
                rows.append(dict(
                    G_mode=mode, algorithm=algo_name, lambda_tv=lam,
                    Q=G.shape[0], mse=mse, mae=mae, tv_of_solution=tv_term,
                    wall_time=wall,
                ))
            logger.info(
                f"  {mode:6s} lam={lam:.0e}: Alg3 mse={mse3:.5g}  Alg5 mse={mse5:.5g}"
            )
    df = pd.DataFrame(rows)
    out_csv = os.path.join(RESULTS, "sec61_regularizer_sweep.csv")
    if os.path.exists(out_csv):
        # Merge with an earlier grid, newest run winning on duplicates, so the
        # sweep can be widened incrementally without redoing what is done.
        prev = pd.read_csv(out_csv)
        df = (
            pd.concat([prev, df], ignore_index=True)
            .drop_duplicates(subset=["G_mode", "algorithm", "lambda_tv"], keep="last")
            .sort_values(["G_mode", "algorithm", "lambda_tv"])
            .reset_index(drop=True)
        )
    df.to_csv(out_csv, index=False)

    best = df.loc[df.groupby(["G_mode", "algorithm"])["mse"].idxmin()]
    for _, r in best.iterrows():
        grid = sorted(df[(df.G_mode == r.G_mode) & (df.algorithm == r.algorithm)]
                      ["lambda_tv"].unique())
        if r.lambda_tv in (grid[0], grid[-1]):
            logger.warning(
                f"  optimum for {r.G_mode}/{r.algorithm} is at lambda={r.lambda_tv:.0e}, "
                f"an EDGE of the swept grid [{grid[0]:.0e}, {grid[-1]:.0e}]. Widen the "
                "grid in that direction before treating it as an optimum."
            )
    best.to_csv(os.path.join(RESULTS, "sec61_best_lambda.csv"), index=False)
    logger.info("Best lambda per (operator, algorithm):\n" + best.to_string(index=False))

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, algo_name in zip(axs, ("Alg3", "Alg5")):
        for mode in MODES:
            sub = df[(df.G_mode == mode) & (df.algorithm == algo_name)]
            ax.plot(sub["lambda_tv"], sub["mse"], marker="o", label=MODE_LABEL[mode])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\lambda_{TV}$")
        ax.set_title(f"{algo_name}: reconstruction MSE vs TV weight")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    axs[0].set_ylabel("MSE")
    fig.tight_layout()
    fig.savefig(os.path.join(VISUALS, "sec61_regularizer_sweep.pdf"))
    fig.savefig(os.path.join(VISUALS, "sec61_regularizer_sweep.png"), dpi=140)
    plt.close(fig)
    return df


# ===========================================================================
# Sec. 6.2: mesh sweep
# ===========================================================================


def study_mesh(logger, deltas=(10, 20), iters=3000, lambda_by_mode=None):
    """TV-regularized Algorithm 5 across refinements, with each operator at
    its own best lambda (passed in from the Sec. 6.1 sweep, since the optimum
    is not transplantable between the two weightings)."""
    logger.info("=== Sec. 6.2: mesh sweep, proxy vs FE jump operator ===")
    lambda_by_mode = lambda_by_mode or {"proxy": 1e-3, "fe_tv": 1e-3}
    rows = []
    for delta in deltas:
        path = os.path.join(SWEEP_ROOT, f"delta{delta}")
        if not os.path.isdir(path):
            continue
        for mode in MODES:
            pb = load_problem(path, G_mode=mode)
            G, lam = pb.G, lambda_by_mode[mode]
            normG = power_iteration_operator_norm(
                lambda v: G @ v, lambda w: G.conj().T @ w, dim=pb.P
            )
            normC = power_iteration_operator_norm(
                lambda v: pb.C @ v, lambda w: pb.C_star @ w, dim=pb.L
            )
            l3 = l_operator_norm_algorithm3(pb, G=G)
            k5 = k_operator_norm_algorithm5(pb, G=G)
            projector = make_projector(pb)
            x5, a5, w5 = run_alg5(pb, G, lam, 0.9 / k5, 0.9 / k5, 0.9 / k5, iters,
                                  projector=projector, name=f"Alg5-{mode}-d{delta}")
            mse5, mae5 = mse_mae(x5, pb.m, pb.P)
            rows.append(dict(
                delta=delta, G_mode=mode, lambda_tv=lam, P=pb.P, Q=G.shape[0],
                normG=normG, normC=normC, l3=l3, k5=k5,
                tau3=0.9 / l3, tau5=0.9 / k5,
                objective=float(a5.f_values[a5.iteration]), mse=mse5, mae=mae5,
                wall_time=w5,
            ))
            logger.info(
                f"  delta={delta} {mode:6s}: Q={G.shape[0]} ||G||={normG:.4f} "
                f"||L3||={l3:.4f} ||K5||={k5:.4f} tau5={0.9/k5:.4f} mse={mse5:.5g}"
            )
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "sec62_mesh_sweep.csv"), index=False)

    fig, axs = plt.subplots(1, 3, figsize=(18, 4.8))
    for mode in MODES:
        sub = df[df.G_mode == mode]
        axs[0].plot(sub["delta"], sub["normG"], marker="o", label=MODE_LABEL[mode])
        axs[1].plot(sub["delta"], sub["tau5"], marker="o", label=MODE_LABEL[mode])
        axs[2].plot(sub["delta"], sub["mse"], marker="o", label=MODE_LABEL[mode])
    axs[0].set_ylabel(r"$\|G\|$")
    axs[0].set_yscale("log")
    axs[0].set_title(r"$\|G\|$ under refinement")
    axs[1].set_ylabel(r"$\tau_5 = 0.9/\|K\|$")
    axs[1].set_title("Algorithm 5 step size")
    axs[2].set_ylabel("MSE")
    axs[2].set_yscale("log")
    axs[2].set_title(f"Reconstruction error ({iters} iters)")
    for ax in axs:
        ax.set_xlabel(r"mesh density $\delta$")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(VISUALS, "sec62_mesh_sweep.pdf"))
    fig.savefig(os.path.join(VISUALS, "sec62_mesh_sweep.png"), dpi=140)
    plt.close(fig)
    return df


# ===========================================================================
# Sec. 6.3: is the optimal lambda transferable across meshes?
# ===========================================================================


def study_lambda_transfer(logger, deltas=(10, 20), iters=3000,
                          lambda_grid=(1e-4, 1e-3, 1e-2, 1e-1)):
    """The practical pay-off of getting the weights right: does the tuned
    `lambda_TV` survive a change of mesh?

    The proxy's `||G m||_1` diverges like h^-1 (measured: 765.8 -> 1730.6 ->
    3667.5 across delta = 10, 20, 40), so holding lambda fixed while refining
    silently doubles the effective penalty, and the optimum must move like h
    to compensate. The |E|-weighted operator's `||G m||_1` converges
    (77.5 -> 81.8 -> 83.9), so its optimum should stay put. That is the
    difference between a regularization weight that means something physical
    (a length) and one that counts mesh entities.

    Algorithm 5 only. `lambda_grid` may be a sequence (shared by both
    operators) or a dict keyed by G_mode: the two optima sit a decade apart,
    and the shift being tested for is smaller than a decade. The proxy's
    ||G m||_1 grows by 2.26x from delta=10 to 20 and the FE operator's by
    1.06x, so a decade-spaced grid cannot resolve it: a per-operator grid
    centred on each optimum is required for the test to conclude anything.
    """
    logger.info("=== Sec. 6.3: is the optimal lambda transferable across meshes? ===")
    rows = []
    for delta in deltas:
        path = os.path.join(SWEEP_ROOT, f"delta{delta}")
        if not os.path.isdir(path):
            continue
        for mode in MODES:
            pb = load_problem(path, G_mode=mode)
            G = pb.G
            k5 = k_operator_norm_algorithm5(pb, G=G)
            projector = make_projector(pb)
            tv_true = float(np.abs(G @ pb.m).sum())
            grid = (
                lambda_grid[mode] if isinstance(lambda_grid, dict) else lambda_grid
            )
            for lam in grid:
                x, algo, wall = run_alg5(
                    pb, G, lam, 0.9 / k5, 0.9 / k5, 0.9 / k5, iters,
                    projector=projector, name=f"Alg5-{mode}-d{delta}-{lam:.0e}",
                )
                mse, mae = mse_mae(x, pb.m, pb.P)
                rows.append(dict(delta=delta, G_mode=mode, lambda_tv=lam,
                                 tv_of_truth=tv_true, mse=mse, mae=mae,
                                 wall_time=wall))
            best = min(
                (r for r in rows if r["delta"] == delta and r["G_mode"] == mode),
                key=lambda r: r["mse"],
            )
            logger.info(
                f"  delta={delta} {mode:6s}: |G m_true|_1={tv_true:8.2f}, "
                f"best lambda={best['lambda_tv']:.0e} (mse={best['mse']:.5g})"
            )
    df = pd.DataFrame(rows)
    out_csv = os.path.join(RESULTS, "sec63_lambda_transfer.csv")
    if os.path.exists(out_csv):
        prev = pd.read_csv(out_csv)
        df = (
            pd.concat([prev, df], ignore_index=True)
            .drop_duplicates(subset=["G_mode", "delta", "lambda_tv"], keep="last")
            .sort_values(["G_mode", "delta", "lambda_tv"])
            .reset_index(drop=True)
        )
    df.to_csv(out_csv, index=False)

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, mode in zip(axs, MODES):
        for delta, sub in df[df.G_mode == mode].groupby("delta"):
            sub = sub.sort_values("lambda_tv")
            ax.plot(sub["lambda_tv"], sub["mse"], marker="o", label=rf"$\delta={delta}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\lambda_{TV}$")
        ax.set_title(MODE_LABEL[mode])
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    axs[0].set_ylabel("MSE")
    fig.suptitle("Does the tuned $\\lambda$ survive a change of mesh?", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(VISUALS, "sec63_lambda_transfer.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(VISUALS, "sec63_lambda_transfer.png"), dpi=140,
                bbox_inches="tight")
    plt.close(fig)
    return df


# ===========================================================================
# Sec. 9: noise robustness
# ===========================================================================


def study_noise(logger, data_path="data", noise_levels=(0.0, 1e-2, 5e-2, 1e-1),
                samples=3, iters=1500, mu=1e-6, lambda_by_mode=None, seed=42):
    """Noise robustness of the TV path under each operator, against the two
    G-independent references (C-NAGD and Tikhonov-regularized Algorithm 5).
    The references are run once: they do not involve `G` at all, and the noise
    realizations are regenerated from the same seed for every arm, so the
    comparison is paired sample by sample."""
    logger.info("=== Sec. 9: noise robustness, proxy vs FE jump operator ===")
    lambda_by_mode = lambda_by_mode or {"proxy": 1e-3, "fe_tv": 1e-3}
    pbs = {mode: load_problem(data_path, G_mode=mode) for mode in MODES}
    pb0 = pbs["proxy"]
    norms = {
        mode: k_operator_norm_algorithm5(pbs[mode], G=pbs[mode].G) for mode in MODES
    }
    k5_tik = k_operator_norm_algorithm5(pb0, G=None)
    projector = make_projector(pb0)

    rows = []
    for sigma_noise in noise_levels:
        # Same seed per noise level, so every arm sees the identical draws.
        rng = np.random.default_rng(seed)
        for sample in range(samples):
            d_list = [
                d_i + sigma_noise * (
                    rng.normal(size=d_i.shape) + 1j * rng.normal(size=d_i.shape)
                )
                for d_i in pb0.d_list
            ]

            pb_noisy = _with_data(pb0, d_list)
            algo = NesterovAcceleratedGradientDescent(
                exp_name="fe_jump", algo_plot_name="C-NAGD",
                f=get_J_3(pb_noisy, mu), df=get_dJ_3(pb_noisy, mu),
                K=get_K_J_3(pb_noisy, mu),
            )
            m_nagd = algo.run(
                x0=np.zeros(pb0.P, dtype=complex), max_iterations=iters,
                store_history=False,
            )
            mse, mae = mse_mae(m_nagd, pb0.m, pb0.P)
            rows.append(dict(sigma_noise=sigma_noise, sample=sample,
                             arm="C-NAGD (Tikhonov)", G_mode="n/a", mse=mse, mae=mae))

            algo_tik = ProjectedChambollePock(
                exp_name="fe_jump", algo_plot_name="Alg5-Tikhonov",
                f=data_fidelity(pb0, d_list), C=pb0.C, d=d_list,
                I=pb0.I, L=pb0.L, P=pb0.P, tau=0.9 / k5_tik,
                sigma_dat=0.9 / k5_tik, sigma_reg=0.9 / k5_tik,
                projector=projector, reg_mode="tikhonov", mu=mu,
            )
            x = algo_tik.run(
                x0=np.zeros(pb0.I * pb0.L + pb0.P, dtype=complex),
                max_iterations=iters, store_history=False,
            )
            mse, mae = mse_mae(x, pb0.m, pb0.P)
            rows.append(dict(sigma_noise=sigma_noise, sample=sample,
                             arm="Alg5-Tikhonov", G_mode="n/a", mse=mse, mae=mae))

            for mode in MODES:
                pb = pbs[mode]
                k5 = norms[mode]
                x, _, _ = run_alg5(
                    pb, pb.G, lambda_by_mode[mode], 0.9 / k5, 0.9 / k5, 0.9 / k5,
                    iters, d_list=d_list, projector=projector,
                    name=f"Alg5-TV-{mode}",
                )
                mse, mae = mse_mae(x, pb.m, pb.P)
                rows.append(dict(sigma_noise=sigma_noise, sample=sample,
                                 arm="Alg5-TV", G_mode=mode, mse=mse, mae=mae))
        logger.info(f"  sigma={sigma_noise}: done ({samples} samples)")

    df = pd.DataFrame(rows)
    df["label"] = np.where(df.G_mode == "n/a", df.arm, df.arm + " (" + df.G_mode + ")")
    df.to_csv(os.path.join(RESULTS, "sec9_noise_raw.csv"), index=False)
    summary = df.groupby(["label", "sigma_noise"])[["mse", "mae"]].agg(["mean", "std"])
    summary.to_csv(os.path.join(RESULTS, "sec9_noise_summary.csv"))
    logger.info("Noise summary (MSE):\n" + summary["mse"].to_string())

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, sub in df.groupby("label"):
        g = sub.groupby("sigma_noise")["mse"]
        ax.errorbar(g.mean().index, g.mean(), yerr=g.std(), marker="o",
                    capsize=3, label=label)
    ax.set_xlabel(r"noise level $\sigma$")
    ax.set_ylabel("MSE")
    ax.set_yscale("log")
    ax.set_title(f"Noise robustness ({samples} samples, {iters} iterations)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(VISUALS, "sec9_noise.pdf"))
    fig.savefig(os.path.join(VISUALS, "sec9_noise.png"), dpi=140)
    plt.close(fig)
    return df, summary


def _with_data(pb, d_list):
    """Copy of `pb` carrying noisy measurements (dataclasses.replace would
    also do, but this keeps the explicit field list of the original code)."""
    import dataclasses

    return dataclasses.replace(pb, d_list=d_list, d=np.concatenate(d_list, axis=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="block,reg,mesh,transfer,noise")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument(
        "--lambdas",
        default="1e-7,1e-6,1e-5,1e-4,1e-3,1e-2",
        help="lambda_TV grid for the regularizer sweep; merges into any existing CSV.",
    )
    ap.add_argument(
        "--deltas",
        default="10,20",
        help=(
            "mesh densities for the iterative studies (block, mesh). Defaults to "
            "10,20: delta=40 (P=6720, L=3461) needs the matrix-free projector and "
            "runs for tens of minutes per arm, so it is opt-in. The *validation* "
            "sweep in scripts/validate_fe_jump_operator.py always covers 40, since "
            "operator norms and TV values need no iterations."
        ),
    )
    args = ap.parse_args()
    which = set(args.only.split(","))
    deltas = tuple(int(d) for d in args.deltas.split(","))
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(VISUALS, exist_ok=True)
    logger = setup_logger(
        name="iwp", log_file=os.path.join(RESULTS, "fe_jump_study.log"),
        level="INFO", log_to_console=True,
    )

    iters_reg = 2000 if args.quick else 10000
    iters_block = 1000 if args.quick else 3000
    samples = 2 if args.quick else 3
    iters_noise = 500 if args.quick else 1500

    # The mesh and noise studies must each use the lambda that is actually best
    # for their operator (the two differ by a decade). Take it from the sweep
    # when it runs in this process, otherwise from the sweep's CSV, so the
    # stages can be run in separate invocations without silently reverting to
    # a placeholder.
    best_lambda = {"proxy": 1e-3, "fe_tv": 1e-3}
    reg_csv = os.path.join(RESULTS, "sec61_regularizer_sweep.csv")
    if "reg" not in which and os.path.exists(reg_csv):
        prev = pd.read_csv(reg_csv)
        prev = prev[prev.algorithm == "Alg5"]
        found = {
            mode: float(
                prev[prev.G_mode == mode].sort_values("mse")["lambda_tv"].iloc[0]
            )
            for mode in MODES
            if (prev.G_mode == mode).any()
        }
        best_lambda.update(found)
        logger.info(f"Best lambda read from {reg_csv}: {best_lambda}")
    if "block" in which:
        study_block(logger, deltas=deltas, iters=iters_block)
    if "reg" in which:
        df = study_regularizer(
            logger, iters=iters_reg,
            lambda_grid=tuple(float(x) for x in args.lambdas.split(",")),
        )
        sub = df[df.algorithm == "Alg5"]
        best_lambda = {
            mode: float(sub[sub.G_mode == mode].sort_values("mse")["lambda_tv"].iloc[0])
            for mode in MODES
        }
        logger.info(f"Best lambda carried into the mesh/noise studies: {best_lambda}")
    if "mesh" in which:
        study_mesh(logger, deltas=deltas, iters=iters_block,
                   lambda_by_mode=best_lambda)
    if "transfer" in which:
        study_lambda_transfer(logger, deltas=deltas, iters=iters_block)
    if "noise" in which:
        study_noise(logger, samples=samples, iters=iters_noise,
                    lambda_by_mode=best_lambda)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
