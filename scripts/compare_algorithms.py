#!/usr/bin/env python
"""Compare the Part 4 (Distributed Optimization) / Part 5 (projection-based
Chambolle-Pock) algorithms of the follow-up report against the baselines of
the original internship report (P-ClosedForm, C-NAGD, FISTA), reproducing
Matthieu Merigot-Lombard's experimental protocol and extending it with the
new algorithms:

  - ChambollePock              (Algorithm 3, Sec. 4.7)
  - DistributedChambollePock   (Algorithm 4, Sec. 4.7, exact consensus)
  - ProjectedChambollePock     (Algorithm 5, Sec. 5.5, exact SMW or inexact
                                 matrix-free CG projection)

as well as the `AffineConstraintProjector` backends compared in Table 2 of
Sec. 5.8-5.9 (S1 "spsolve", S2 "cached_splu", S3 "smw", S4 "smw_cg").

Run:
    pixi run python scripts/compare_algorithms.py [--quick] [--exp-name NAME]

Outputs are written under runs/<exp-name>/{visuals,results}, following the
same convention as `iwp.main`. This module is also meant to be imported
piece by piece from the companion notebook
(`part4_5_chambolle_pock_comparison.ipynb`) so that both artifacts share a
single, tested implementation of every experiment.
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

from iwp.algorithms.algorithms import (
    FISTA,  # noqa: E402
    AffineConstraintProjector,
    ChambollePock,
    ClosedFormSolution,
    DistributedChambollePock,
    NesterovAcceleratedGradientDescent,
    ProjectedChambollePock,
    make_tikhonov_dual_prox,
    make_tv_dual_prox,
)
from iwp.algorithms.plot import plot_all_algorithms_convergence  # noqa: E402
from iwp.data.load_experiment_data import load_experiment_data  # noqa: E402
from iwp.experiments.comparison import (
    ProblemData,  # noqa: E402
    block_step_sizes_algorithm3,
    block_step_sizes_algorithm5,
    get_closed_form_solution_J_1,
    get_dJ_3,
    get_grad_J_2,
    get_J_1,
    get_J_2,
    get_J_3,
    get_K_J_2,
    get_K_J_3,
    get_prox_J_2_spsolve,
    k_operator_norm_algorithm5,
    l_operator_norm_algorithm3,
    load_problem,
    run_and_record,
)
from iwp.utils.logger import setup_logger  # noqa: E402
from iwp.utils.operators import power_iteration_operator_norm  # noqa: E402
from iwp.utils.utils import make_dirs, set_seed  # noqa: E402

SWEEP_ROOT = os.path.join("data", "sweep")


def objective_data_fidelity(pb: ProblemData):
    """`sum_i (1/2)||C u_i - d_i||^2`, the data term common to Algorithms
    3/4/5 (the only term they all actually minimize; TV/Tikhonov enters
    through the dualized/proxed regularizer and is reported separately)."""

    def f(x):
        total = 0.0
        for i in range(pb.I):
            ui = x[i * pb.L : (i + 1) * pb.L]
            r = pb.C @ ui - pb.d_list[i]
            total += 0.5 * np.vdot(r, r).real
        return total

    return f


def mse_mae(x_or_m, m_true, P):
    m_pred = x_or_m[-P:] if x_or_m.shape[0] != P else x_or_m
    mse = float(np.mean(np.abs(m_pred - m_true) ** 2))
    mae = float(np.mean(np.abs(m_pred - m_true)))
    return mse, mae


# ===========================================================================
# Section 1: baselines (P-ClosedForm, C-NAGD, FISTA), Matthieu's exact setup
# ===========================================================================


def run_baselines(
    pb,
    dirs,
    logger,
    lambd=1e-5,
    mu1=1e-7,
    mu2=1e-6,
    mu3=1e-6,
    max_iter_cnagd=5000,
    max_iter_fista=30000,
):
    logger.info("=== Section 1: baselines (P-ClosedForm, C-NAGD, FISTA) ===")
    x0 = np.zeros(pb.I * pb.L + pb.P, dtype=complex)

    J1 = get_J_1(pb, lambd, mu1)
    closed = get_closed_form_solution_J_1(pb, lambd, mu1)
    algo1 = ClosedFormSolution(
        exp_name="part45",
        algo_plot_name="P-ClosedForm",
        f=J1,
        solution=closed,
        logger=logger,
    )
    x1, t1 = run_and_record(algo1, x0, 1, pb.m, dirs["visuals"], dirs["results"])

    J3 = get_J_3(pb, mu3)
    dJ3 = get_dJ_3(pb, mu3)
    K3 = get_K_J_3(pb, mu3)
    algo3 = NesterovAcceleratedGradientDescent(
        exp_name="part45",
        algo_plot_name="C-NAGD",
        f=J3,
        df=dJ3,
        K=K3,
        logger=logger,
    )
    x3, t3 = run_and_record(
        algo3, x0[-pb.P :], max_iter_cnagd, pb.m, dirs["visuals"], dirs["results"]
    )

    J2 = get_J_2(pb, mu2)
    grad2 = get_grad_J_2(pb, mu2)
    prox2 = get_prox_J_2_spsolve(pb)
    K2 = get_K_J_2(pb, mu2)
    algo2 = FISTA(
        exp_name="part45",
        algo_plot_name="FISTA",
        f=J2,
        grad=grad2,
        prox=prox2,
        K=K2,
        logger=logger,
    )
    x2, t2 = run_and_record(
        algo2, x0, max_iter_fista, pb.m, dirs["visuals"], dirs["results"]
    )

    plot_all_algorithms_convergence(
        [algo1, algo3, algo2],
        dirs["visuals"],
        show=False,
        save=True,
        show_time_memory=True,
        results=True,
    )

    algos = {"P-ClosedForm": algo1, "C-NAGD": algo3, "FISTA": algo2}
    rows = []
    for name, algo in algos.items():
        mse, mae = mse_mae(algo.x_values[-1], pb.m, pb.P)
        rows.append(
            dict(
                algorithm=name,
                iterations=algo.iteration,
                time_s=algo.cv_time,
                memory_kb=algo.memory_used / 1024,
                mse=mse,
                mae=mae,
                objective=float(algo.f_values[-1]),
            )
        )
    df = pd.DataFrame(rows)
    df.to_csv(
        os.path.join(dirs["results"], "section1_baselines_summary.csv"), index=False
    )
    logger.info("Baselines summary:\n" + df.to_string(index=False))
    return algos, df


# ===========================================================================
# Section 2: Algorithm 3 (dualized PDE) vs Algorithm 5 (projected PDE),
# both under the same Tikhonov weight as the baselines, at their respective
# theory-derived step sizes tau = sigma = 0.9 / ||operator||.
# ===========================================================================


def run_algorithm3_and_5(pb, dirs, logger, mu=1e-6, max_iterations=30000):
    logger.info("=== Section 2: Algorithm 3 (dualized) vs Algorithm 5 (projected) ===")
    f = objective_data_fidelity(pb)
    x0 = np.zeros(pb.I * pb.L + pb.P, dtype=complex)

    l3 = l_operator_norm_algorithm3(pb, G=None)
    tau3 = sigma3 = 0.9 / l3
    algo3 = ChambollePock(
        exp_name="part45",
        algo_plot_name="Alg3-ChambollePock",
        f=f,
        A=pb.A,
        B=pb.B_list,
        C=pb.C,
        G=None,
        d=pb.d_list,
        I=pb.I,
        L=pb.L,
        P=pb.P,
        tau=tau3,
        sigma_pde=sigma3,
        prox_dual_reg=None,
        logger=logger,
    )
    x3, t3 = run_and_record(
        algo3, x0, max_iterations, pb.m, dirs["visuals"], dirs["results"]
    )

    k5 = k_operator_norm_algorithm5(pb, G=None)
    tau5 = sigma5 = 0.9 / k5
    projector = AffineConstraintProjector(pb.A, pb.B_list, method="smw", logger=logger)
    algo5 = ProjectedChambollePock(
        exp_name="part45",
        algo_plot_name="Alg5-ProjectedCP-SMW",
        f=f,
        C=pb.C,
        d=pb.d_list,
        I=pb.I,
        L=pb.L,
        P=pb.P,
        tau=tau5,
        sigma_dat=sigma5,
        sigma_reg=sigma5,
        projector=projector,
        reg_mode="tikhonov",
        mu=mu,
        logger=logger,
    )
    x5, t5 = run_and_record(
        algo5, x0, max_iterations, pb.m, dirs["visuals"], dirs["results"]
    )

    plot_all_algorithms_convergence(
        [algo3, algo5],
        dirs["visuals"],
        show=False,
        save=True,
        show_time_memory=True,
        results=True,
    )
    # rename combined plot so it doesn't overwrite section 1's
    src = os.path.join(dirs["visuals"], "All_algorithms.pdf")
    dst = os.path.join(dirs["visuals"], "section2_algorithm3_vs_5.pdf")
    if os.path.exists(src):
        os.replace(src, dst)

    rows = []
    for name, algo, norm in [
        ("Alg3-ChambollePock", algo3, l3),
        ("Alg5-ProjectedCP-SMW", algo5, k5),
    ]:
        mse, mae = mse_mae(algo.x_values[-1], pb.m, pb.P)
        rows.append(
            dict(
                algorithm=name,
                operator_norm=norm,
                tau=tau3 if "3" in name else tau5,
                iterations=algo.iteration,
                time_s=algo.cv_time,
                memory_kb=algo.memory_used / 1024,
                mse=mse,
                mae=mae,
                final_residual=algo.current_residual,
            )
        )
    df = pd.DataFrame(rows)
    df.to_csv(
        os.path.join(dirs["results"], "section2_algorithm3_vs_5_summary.csv"),
        index=False,
    )
    logger.info("Algorithm 3 vs 5 summary:\n" + df.to_string(index=False))
    logger.info(
        f"Key finding: ||L|| (Alg. 3, dualized PDE) = {l3:.3f} is dominated by "
        f"||A|| ({power_iteration_operator_norm(lambda v: pb.A @ v, lambda w: pb.A_star @ w, pb.L):.3f}); "
        f"||K|| (Alg. 5, projected PDE) = {k5:.3f} never involves A "
        f"and is dominated by ||C||, giving a {l3 / k5:.2f}x larger admissible step size."
    )
    return {"Alg3": algo3, "Alg5": algo5}, df


# ===========================================================================
# Section 3: Distributed CP (Algorithm 4) vs centralized (Algorithm 3):
# exact-consensus correctness, and the regularization-weight-vs-S subtlety.
# ===========================================================================


def run_distributed_comparison(pb, dirs, logger, mu=1e-6, S=2, max_iterations=10000):
    logger.info(
        "=== Section 3: Distributed (Algorithm 4) vs centralized (Algorithm 3) ==="
    )
    f = objective_data_fidelity(pb)
    x0 = np.zeros(pb.I * pb.L + pb.P, dtype=complex)
    l3 = l_operator_norm_algorithm3(pb, G=None)
    tau = sigma = 0.9 / l3
    Id = sp.eye(pb.P, format="csr")

    algo_c = ChambollePock(
        exp_name="part45",
        algo_plot_name="Alg3-Centralized",
        f=f,
        A=pb.A,
        B=pb.B_list,
        C=pb.C,
        G=Id,
        d=pb.d_list,
        I=pb.I,
        L=pb.L,
        P=pb.P,
        tau=tau,
        sigma_pde=sigma,
        prox_dual_reg=make_tikhonov_dual_prox(mu),
        logger=logger,
    )
    x_c, _ = run_and_record(
        algo_c, x0, max_iterations, pb.m, dirs["visuals"], dirs["results"]
    )

    agent_indices = [[i for i in range(s, pb.I, S)] for s in range(S)]
    variants = {"unscaled_mu": mu, "scaled_mu_over_S": mu / S}
    rows = []
    algos = {"Alg3-Centralized": algo_c}
    for label, mu_variant in variants.items():
        algo_d = DistributedChambollePock(
            exp_name="part45",
            algo_plot_name=f"Alg4-Distributed-{label}",
            f=f,
            A=pb.A,
            B=pb.B_list,
            C=pb.C,
            G=Id,
            d=pb.d_list,
            S=S,
            I=pb.I,
            L=pb.L,
            P=pb.P,
            tau=tau,
            sigma_pde=sigma,
            prox_dual_reg=make_tikhonov_dual_prox(mu_variant),
            agent_indices=agent_indices,
            use_mpi=False,
            logger=logger,
        )
        x_d, _ = run_and_record(
            algo_d, x0, max_iterations, pb.m, dirs["visuals"], dirs["results"]
        )
        algos[f"Alg4-Distributed-{label}"] = algo_d
        mse, mae = mse_mae(x_d, pb.m, pb.P)
        rows.append(
            dict(
                variant=label,
                mu_used=mu_variant,
                iterations=algo_d.iteration,
                time_s=algo_d.cv_time,
                mse=mse,
                mae=mae,
                l2_dist_to_centralized=float(np.linalg.norm(x_d - x_c)),
            )
        )
    df = pd.DataFrame(rows)
    df.to_csv(
        os.path.join(dirs["results"], "section3_distributed_vs_centralized.csv"),
        index=False,
    )
    logger.info("Distributed vs centralized summary:\n" + df.to_string(index=False))
    logger.info(
        "Key finding: Eq. (35) sums the regularizer once per agent with no 1/S "
        "factor, so DistributedChambollePock must be given mu/S to reproduce "
        "the centralized ChambollePock solution -- see l2_dist_to_centralized "
        "above (unscaled vs. scaled)."
    )
    return algos, df


# ===========================================================================
# Section 4: step-size sensitivity (Eq. 31: tau*sigma*||op||^2 < 1)
# ===========================================================================


def run_step_size_sensitivity(
    pb,
    dirs,
    logger,
    mu=1e-6,
    max_iterations=3000,
    alphas=(0.3, 0.5, 0.9, 0.99, 1.0, 1.05, 1.2),
):
    logger.info("=== Section 4: step-size (tau, sigma) sensitivity ===")
    f = objective_data_fidelity(pb)
    x0 = np.zeros(pb.I * pb.L + pb.P, dtype=complex)
    l3 = l_operator_norm_algorithm3(pb, G=None)
    k5 = k_operator_norm_algorithm5(pb, G=None)

    rows = []
    curves = {"Alg3": {}, "Alg5": {}}
    for alpha in alphas:
        tau3 = sigma3 = alpha / l3
        algo3 = ChambollePock(
            exp_name="part45",
            algo_plot_name=f"Alg3-alpha{alpha}",
            f=f,
            A=pb.A,
            B=pb.B_list,
            C=pb.C,
            G=None,
            d=pb.d_list,
            I=pb.I,
            L=pb.L,
            P=pb.P,
            tau=tau3,
            sigma_pde=sigma3,
            prox_dual_reg=None,
        )
        try:
            x3 = algo3.run(x0=x0, max_iterations=max_iterations)
            diverged3 = (
                not np.all(np.isfinite(x3))
                or algo3.f_values[-1] > 10 * algo3.f_values[0] + 1e3
            )
        except Exception:
            diverged3 = True
        curves["Alg3"][alpha] = algo3.f_values.copy()
        rows.append(
            dict(
                algorithm="Alg3",
                alpha=alpha,
                tau=tau3,
                final_objective=float(algo3.f_values[-1]),
                diverged=diverged3,
            )
        )

        projector = AffineConstraintProjector(pb.A, pb.B_list, method="smw")
        tau5 = sigma5 = alpha / k5
        algo5 = ProjectedChambollePock(
            exp_name="part45",
            algo_plot_name=f"Alg5-alpha{alpha}",
            f=f,
            C=pb.C,
            d=pb.d_list,
            I=pb.I,
            L=pb.L,
            P=pb.P,
            tau=tau5,
            sigma_dat=sigma5,
            sigma_reg=sigma5,
            projector=projector,
            reg_mode="tikhonov",
            mu=mu,
        )
        try:
            x5 = algo5.run(x0=x0, max_iterations=max_iterations)
            diverged5 = (
                not np.all(np.isfinite(x5))
                or algo5.f_values[-1] > 10 * algo5.f_values[0] + 1e3
            )
        except Exception:
            diverged5 = True
        curves["Alg5"][alpha] = algo5.f_values.copy()
        rows.append(
            dict(
                algorithm="Alg5",
                alpha=alpha,
                tau=tau5,
                final_objective=float(algo5.f_values[-1]),
                diverged=diverged5,
            )
        )

    df = pd.DataFrame(rows)
    df.to_csv(
        os.path.join(dirs["results"], "section4_stepsize_sensitivity.csv"), index=False
    )

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    for ax, key, norm_name in zip(axs, ["Alg3", "Alg5"], ["||L||", "||K||"]):
        for alpha, fvals in curves[key].items():
            fvals_plot = np.clip(fvals, 1e-12, 1e12)
            ax.plot(fvals_plot, label=f"alpha={alpha}")
        ax.set_yscale("log")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Objective function")
        ax.set_title(f"{key}: tau=sigma=alpha/{norm_name}")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(dirs["visuals"], "section4_stepsize_sensitivity.pdf"))
    plt.close()

    logger.info("Step-size sensitivity summary:\n" + df.to_string(index=False))
    return df, curves


# ===========================================================================
# Section 4b: block dual step sizes -- one sigma per dualized block instead
# of a single scalar shared by both (Eq. 34 for Algorithm 3/4, Eq. 56 for
# Algorithm 5).
# ===========================================================================


def _tv_energy(m, G, lambda_tv):
    """`lambda_tv * ||G m||_{2,1}`, the regularizer the algorithms actually
    minimize alongside the data term (with singleton groups, Sec. 5.2, the
    group l2,1 norm is the plain l1 norm of the jumps)."""
    return float(lambda_tv * np.sum(np.abs(G @ m)))


def run_block_sigma_comparison(
    pb,
    dirs,
    logger,
    lambda_tv=1e-3,
    S=2,
    gammas=(0.3, 1.0, 3.0, 10.0),
    max_iterations=3000,
    safety=0.9,
):
    """Compare the scalar dual metric (`sigma_pde = sigma_reg`, Eq. (31)/(33))
    against the block-diagonal one (`Sigma = diag(sigma_pde I, sigma_reg I)`,
    Eq. (34)/(56)) on all three Chambolle-Pock instantiations, with Total
    Variation enabled so that the regularizer block is actually present and
    its `O(1/h)` scale differs from the other block's.

    Controlled ablation. Both metrics are parametrized by the *same* knob
    `gamma` (the primal/dual ratio) and placed at the *same* realized margin
    `safety`, so that the only difference left between a `scalar_gammaX` and
    the matching `block_gammaX` row is how the dual budget is split between
    the two blocks:

        scalar:  tau = safety/gamma,  sigma = gamma/||.||^2       (both blocks)
        block:   tau ~ safety/gamma,  sigma_b = gamma/||._b||^2   (per block)

    Both realize `condition_lhs = safety` (recorded per row). The scalar
    parametrization is the one used in Sections 2/4 up to a reparametrization:
    `gamma = sqrt(safety)*||.||` gives back `tau = sigma = sqrt(safety)/||.||`,
    so the sweep brackets that convention rather than replacing it.

    What to expect (Sec. 4.7 vs 5.6). For Algorithm 5 the block metric is a
    genuine win: both `||C||` and `||G||` are moderate, so `sigma_dat` can be
    raised far above `sigma_reg`. For Algorithm 3/4 the gain is bounded: the
    PDE block keeps `||A||` inside `L`, so `sigma_pde <~ 1/(tau ||A||^2)`
    however the metric is balanced, and only the TV block is relaxed.
    """
    logger.info("=== Section 4b: block (per-dualized-block) dual step sizes ===")
    f = objective_data_fidelity(pb)
    x0 = np.zeros(pb.I * pb.L + pb.P, dtype=complex)
    G = pb.G
    agent_indices = [[i for i in range(s, pb.I, S)] for s in range(S)]

    l3 = l_operator_norm_algorithm3(pb, G=G)
    k5 = k_operator_norm_algorithm5(pb, G=G)

    # (label, metric, gamma, tau, sigma_block1, sigma_reg, condition LHS).
    configs3, configs5, block_info = [], [], []
    for gamma in gammas:
        label_s, label_b = f"scalar_gamma{gamma:g}", f"block_gamma{gamma:g}"
        # Scalar metric: a single sigma for both blocks, necessarily dictated
        # by the larger of the two block norms (which is what ||L||/||K|| is).
        sc3, sc5 = gamma / l3**2, gamma / k5**2
        configs3.append((label_s, "scalar", gamma, safety / gamma, sc3, sc3, safety))
        configs5.append((label_s, "scalar", gamma, safety / gamma, sc5, sc5, safety))

        s3 = block_step_sizes_algorithm3(pb, G=G, gamma=gamma, safety=safety)
        s5 = block_step_sizes_algorithm5(pb, G=G, gamma=gamma, safety=safety)
        configs3.append(
            (
                label_b,
                "block",
                gamma,
                s3["tau"],
                s3["sigma_pde"],
                s3["sigma_reg"],
                s3["condition_lhs"],
            )
        )
        configs5.append(
            (
                label_b,
                "block",
                gamma,
                s5["tau"],
                s5["sigma_dat"],
                s5["sigma_reg"],
                s5["condition_lhs"],
            )
        )
        block_info.append(dict(gamma=gamma, **{f"alg3_{k}": v for k, v in s3.items()}))
        block_info[-1].update({f"alg5_{k}": v for k, v in s5.items()})

    if block_info:
        b0 = block_info[0]
        logger.info(
            f"Block norms -- Alg3: ||L_pde||={b0['alg3_norm_pde']:.3f}, "
            f"||L_tv||={b0['alg3_norm_reg']:.3f} "
            f"(ratio {b0['alg3_norm_pde'] / b0['alg3_norm_reg']:.2f}); "
            f"Alg5: ||K_dat||={b0['alg5_norm_dat']:.3f}, "
            f"||K_tv||={b0['alg5_norm_reg']:.3f} "
            f"(ratio {b0['alg5_norm_dat'] / b0['alg5_norm_reg']:.2f}). "
            f"Scalar norms: ||L||={l3:.3f}, ||K||={k5:.3f}"
        )

    rows = []
    curves = {"Alg3": {}, "Alg4": {}, "Alg5": {}}

    def record(family, label, metric, gamma, algo, x, tau, sigma_a, sigma_b, lhs):
        finite = bool(np.all(np.isfinite(x)))
        m_hat = x[-pb.P :]
        mse, mae = mse_mae(x, pb.m, pb.P) if finite else (np.nan, np.nan)
        data_fit = float(algo.f_values[-1]) if finite else np.nan
        tv = _tv_energy(m_hat, G, lambda_tv) if finite else np.nan
        curves[family][label] = algo.f_values.copy()
        rows.append(
            dict(
                algorithm=family,
                variant=label,
                metric=metric,
                gamma=gamma,
                tau=tau,
                sigma_pde_or_dat=sigma_a,
                sigma_reg=sigma_b,
                sigma_ratio=sigma_b / sigma_a,
                condition_lhs=lhs,
                iterations=algo.iteration,
                time_s=algo.cv_time,
                data_fidelity=data_fit,
                tv_energy=tv,
                total_objective=data_fit + tv if finite else np.nan,
                mse=mse,
                mae=mae,
                feasibility=float(np.linalg.norm(pb.E @ x)) if finite else np.nan,
                diverged=not finite,
            )
        )

    for label, metric, gamma, tau, sigma_pde, sigma_reg, lhs in configs3:
        algo3 = ChambollePock(
            exp_name="part45",
            algo_plot_name=f"Alg3-TV-{label}",
            f=f,
            A=pb.A,
            B=pb.B_list,
            C=pb.C,
            G=G,
            d=pb.d_list,
            I=pb.I,
            L=pb.L,
            P=pb.P,
            tau=tau,
            sigma_pde=sigma_pde,
            sigma_reg=sigma_reg,
            prox_dual_reg=make_tv_dual_prox(lambda_tv),
        )
        x3 = algo3.run(x0=x0, max_iterations=max_iterations)
        record("Alg3", label, metric, gamma, algo3, x3, tau, sigma_pde, sigma_reg, lhs)

        # Algorithm 4 reuses Algorithm 3's operator/metric; the regularizer
        # weight is divided by S (Eq. (35) sums R once per agent, cf. the
        # `DistributedChambollePock` docstring and Section 3).
        algo4 = DistributedChambollePock(
            exp_name="part45",
            algo_plot_name=f"Alg4-TV-{label}",
            f=f,
            A=pb.A,
            B=pb.B_list,
            C=pb.C,
            G=G,
            d=pb.d_list,
            S=S,
            I=pb.I,
            L=pb.L,
            P=pb.P,
            tau=tau,
            sigma_pde=sigma_pde,
            sigma_reg=sigma_reg,
            prox_dual_reg=make_tv_dual_prox(lambda_tv / S),
            agent_indices=agent_indices,
            use_mpi=False,
        )
        x4 = algo4.run(x0=x0, max_iterations=max_iterations)
        record("Alg4", label, metric, gamma, algo4, x4, tau, sigma_pde, sigma_reg, lhs)

    for label, metric, gamma, tau, sigma_dat, sigma_reg, lhs in configs5:
        projector = AffineConstraintProjector(pb.A, pb.B_list, method="smw")
        algo5 = ProjectedChambollePock(
            exp_name="part45",
            algo_plot_name=f"Alg5-TV-{label}",
            f=f,
            C=pb.C,
            d=pb.d_list,
            I=pb.I,
            L=pb.L,
            P=pb.P,
            tau=tau,
            sigma_dat=sigma_dat,
            sigma_reg=sigma_reg,
            projector=projector,
            reg_mode="tv",
            G=G,
            lambda_tv=lambda_tv,
        )
        x5 = algo5.run(x0=x0, max_iterations=max_iterations)
        record("Alg5", label, metric, gamma, algo5, x5, tau, sigma_dat, sigma_reg, lhs)

    df = pd.DataFrame(rows)
    df.to_csv(
        os.path.join(dirs["results"], "section4b_block_sigma_comparison.csv"),
        index=False,
    )
    norms_df = pd.DataFrame(block_info)
    norms_df.to_csv(
        os.path.join(dirs["results"], "section4b_block_norms.csv"), index=False
    )

    fig, axs = plt.subplots(1, 3, figsize=(18, 4.5))
    colors = {float(g): f"C{j}" for j, g in enumerate(gammas)}
    for ax, family in zip(axs, ["Alg3", "Alg4", "Alg5"]):
        for label, fvals in curves[family].items():
            metric, _, gamma_txt = label.partition("_gamma")
            gamma = float(gamma_txt)
            ax.plot(
                np.clip(fvals, 1e-12, 1e12),
                label=label,
                color=colors.get(gamma),
                linestyle="--" if metric == "scalar" else "-",
                linewidth=1.2 if metric == "scalar" else 2.0,
            )
        ax.set_yscale("log")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Data fidelity")
        ax.set_title(f"{family}: scalar (dashed) vs block (solid)", fontsize=10)
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(dirs["visuals"], "section4b_block_sigma_comparison.pdf"))
    plt.close()

    logger.info("Block vs scalar dual metric summary:\n" + df.to_string(index=False))
    return df, curves, norms_df


# ===========================================================================
# Section 5: Total Variation (graph-gradient proxy G) vs Tikhonov
# ===========================================================================


def run_tv_vs_tikhonov(
    pb,
    dirs,
    logger,
    mu=1e-6,
    lambda_tv_grid=(1e-3, 1e-2, 1e-1, 1.0),
    max_iterations=10000,
):
    logger.info("=== Section 5: Total Variation (graph-gradient proxy) vs Tikhonov ===")
    f = objective_data_fidelity(pb)
    x0 = np.zeros(pb.I * pb.L + pb.P, dtype=complex)
    G = pb.G
    l3_tik = l_operator_norm_algorithm3(pb, G=None)
    l3_tv = l_operator_norm_algorithm3(pb, G=G)
    k5_tik = k_operator_norm_algorithm5(pb, G=None)
    k5_tv = k_operator_norm_algorithm5(pb, G=G)

    rows = []
    algos = {}

    tau3 = sigma3 = 0.9 / l3_tik
    algo3_tik = ChambollePock(
        exp_name="part45",
        algo_plot_name="Alg3-Tikhonov",
        f=f,
        A=pb.A,
        B=pb.B_list,
        C=pb.C,
        G=sp.eye(pb.P, format="csr"),
        d=pb.d_list,
        I=pb.I,
        L=pb.L,
        P=pb.P,
        tau=tau3,
        sigma_pde=sigma3,
        prox_dual_reg=make_tikhonov_dual_prox(mu),
    )
    x, _ = run_and_record(
        algo3_tik, x0, max_iterations, pb.m, dirs["visuals"], dirs["results"]
    )
    mse, mae = mse_mae(x, pb.m, pb.P)
    rows.append(
        dict(algorithm="Alg3", regularizer="tikhonov", weight=mu, mse=mse, mae=mae)
    )
    algos["Alg3-Tikhonov"] = algo3_tik

    tau5 = sigma5 = 0.9 / k5_tik
    projector = AffineConstraintProjector(pb.A, pb.B_list, method="smw")
    algo5_tik = ProjectedChambollePock(
        exp_name="part45",
        algo_plot_name="Alg5-Tikhonov",
        f=f,
        C=pb.C,
        d=pb.d_list,
        I=pb.I,
        L=pb.L,
        P=pb.P,
        tau=tau5,
        sigma_dat=sigma5,
        sigma_reg=sigma5,
        projector=projector,
        reg_mode="tikhonov",
        mu=mu,
    )
    x, _ = run_and_record(
        algo5_tik, x0, max_iterations, pb.m, dirs["visuals"], dirs["results"]
    )
    mse, mae = mse_mae(x, pb.m, pb.P)
    rows.append(
        dict(algorithm="Alg5", regularizer="tikhonov", weight=mu, mse=mse, mae=mae)
    )
    algos["Alg5-Tikhonov"] = algo5_tik

    for lambda_tv in lambda_tv_grid:
        tau3 = sigma3 = 0.9 / l3_tv
        algo3_tv = ChambollePock(
            exp_name="part45",
            algo_plot_name=f"Alg3-TV-lam{lambda_tv:.0e}",
            f=f,
            A=pb.A,
            B=pb.B_list,
            C=pb.C,
            G=G,
            d=pb.d_list,
            I=pb.I,
            L=pb.L,
            P=pb.P,
            tau=tau3,
            sigma_pde=sigma3,
            prox_dual_reg=make_tv_dual_prox(lambda_tv),
        )
        x, _ = run_and_record(
            algo3_tv, x0, max_iterations, pb.m, dirs["visuals"], dirs["results"]
        )
        mse, mae = mse_mae(x, pb.m, pb.P)
        rows.append(
            dict(algorithm="Alg3", regularizer="tv", weight=lambda_tv, mse=mse, mae=mae)
        )
        algos[f"Alg3-TV-lam{lambda_tv:.0e}"] = algo3_tv

        tau5 = sigma5 = 0.9 / k5_tv
        projector = AffineConstraintProjector(pb.A, pb.B_list, method="smw")
        algo5_tv = ProjectedChambollePock(
            exp_name="part45",
            algo_plot_name=f"Alg5-TV-lam{lambda_tv:.0e}",
            f=f,
            C=pb.C,
            d=pb.d_list,
            I=pb.I,
            L=pb.L,
            P=pb.P,
            tau=tau5,
            sigma_dat=sigma5,
            sigma_reg=sigma5,
            projector=projector,
            reg_mode="tv",
            G=G,
            lambda_tv=lambda_tv,
        )
        x, _ = run_and_record(
            algo5_tv, x0, max_iterations, pb.m, dirs["visuals"], dirs["results"]
        )
        mse, mae = mse_mae(x, pb.m, pb.P)
        rows.append(
            dict(algorithm="Alg5", regularizer="tv", weight=lambda_tv, mse=mse, mae=mae)
        )
        algos[f"Alg5-TV-lam{lambda_tv:.0e}"] = algo5_tv

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(dirs["results"], "section5_tv_vs_tikhonov.csv"), index=False)
    logger.info("TV vs Tikhonov summary:\n" + df.to_string(index=False))
    return algos, df


# ===========================================================================
# Section 6: exact (SMW direct) vs inexact (matrix-free CG) projection
# ===========================================================================


def run_exact_vs_inexact_projection(
    pb, dirs, logger, mu=1e-6, max_iterations=8000, cg_gammas=(0.5, 0.8)
):
    logger.info("=== Section 6: exact (SMW) vs inexact (matrix-free CG) projection ===")
    f = objective_data_fidelity(pb)
    x0 = np.zeros(pb.I * pb.L + pb.P, dtype=complex)
    k5 = k_operator_norm_algorithm5(pb, G=None)
    tau = sigma = 0.9 / k5

    rows = []
    algos = {}
    projector_exact = AffineConstraintProjector(pb.A, pb.B_list, method="smw")
    algo_exact = ProjectedChambollePock(
        exp_name="part45",
        algo_plot_name="Alg5-SMW-exact",
        f=f,
        C=pb.C,
        d=pb.d_list,
        I=pb.I,
        L=pb.L,
        P=pb.P,
        tau=tau,
        sigma_dat=sigma,
        sigma_reg=sigma,
        projector=projector_exact,
        reg_mode="tikhonov",
        mu=mu,
    )
    x_exact, t_exact = run_and_record(
        algo_exact, x0, max_iterations, pb.m, dirs["visuals"], dirs["results"]
    )
    mse, mae = mse_mae(x_exact, pb.m, pb.P)
    rows.append(
        dict(
            variant="smw_exact",
            cg_gamma=None,
            time_s=t_exact,
            mean_inner_iters=None,
            mse=mse,
            mae=mae,
            final_residual=algo_exact.current_residual,
        )
    )
    algos["Alg5-SMW-exact"] = algo_exact

    for gamma in cg_gammas:
        projector_cg = AffineConstraintProjector(
            pb.A,
            pb.B_list,
            method="smw_cg",
            cg_gamma=gamma,
            cg_eta0=1.0,
        )
        algo_cg = ProjectedChambollePock(
            exp_name="part45",
            algo_plot_name=f"Alg5-SMW-CG-gamma{gamma}",
            f=f,
            C=pb.C,
            d=pb.d_list,
            I=pb.I,
            L=pb.L,
            P=pb.P,
            tau=tau,
            sigma_dat=sigma,
            sigma_reg=sigma,
            projector=projector_cg,
            reg_mode="tikhonov",
            mu=mu,
        )
        x_cg, t_cg = run_and_record(
            algo_cg, x0, max_iterations, pb.m, dirs["visuals"], dirs["results"]
        )
        mse, mae = mse_mae(x_cg, pb.m, pb.P)
        rows.append(
            dict(
                variant=f"smw_cg_gamma{gamma}",
                cg_gamma=gamma,
                time_s=t_cg,
                mean_inner_iters=float(np.mean(projector_cg.inner_iterations)),
                mse=mse,
                mae=mae,
                final_residual=algo_cg.current_residual,
            )
        )
        algos[f"Alg5-SMW-CG-gamma{gamma}"] = algo_cg

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(projector_cg.inner_iterations)
        ax.set_xlabel("Outer iteration")
        ax.set_ylabel("Inner CG iterations")
        ax.set_title(f"CG inner-iteration count (gamma={gamma}), warm-started")
        plt.tight_layout()
        plt.savefig(
            os.path.join(dirs["visuals"], f"section6_cg_inner_iters_gamma{gamma}.pdf")
        )
        plt.close()

    df = pd.DataFrame(rows)
    df.to_csv(
        os.path.join(dirs["results"], "section6_exact_vs_inexact.csv"), index=False
    )
    logger.info("Exact vs inexact projection summary:\n" + df.to_string(index=False))
    return algos, df


# ===========================================================================
# Section 7: projector backend benchmark (S1/S2/S3/S4), Table 2 reproduction
# ===========================================================================


class _HardTimeout(Exception):
    pass


def _with_hard_timeout(seconds, fn, *fn_args, **fn_kwargs):
    """Best-effort SIGALRM-based hard timeout around `fn`, used as a safety
    net around the "spsolve"/"cached_splu" projector backends: their setup
    cost is a *single* factorization of the I*L x I*L matrix E E*, which
    Sec. 5.3 warns grows with O(I^2 L) fill and is "ordering-dependent" --
    i.e. can blow up unpredictably rather than gracefully. Without this, one
    oversized configuration could hang the whole sweep."""
    import signal

    def _handler(signum, frame):
        raise _HardTimeout(f"exceeded {seconds}s")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(int(seconds))
    try:
        return fn(*fn_args, **fn_kwargs)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def _benchmark_projector(A, B_list, method, n_calls=10, time_budget_s=8.0, **kwargs):
    """Time `AffineConstraintProjector(method=...)`'s setup and per-call
    projection cost. The "spsolve" backend re-forms `E E*` from scratch on
    every call (Sec. 5.3's point that this becomes ~O(I^2 L) dense-ish for
    many sources), so at large I/L a fixed `n_calls` can become very slow;
    `time_budget_s` adaptively caps the number of timed calls (down to a
    minimum of 1) so the benchmark stays bounded regardless of problem size,
    at the cost of a noisier per-call estimate for the largest problems."""
    I = len(B_list)
    L = A.shape[0]
    P = B_list[0].shape[1]
    rng = np.random.default_rng(0)
    u_list = [rng.normal(size=L) + 1j * rng.normal(size=L) for _ in range(I)]
    m = rng.normal(size=P) + 1j * rng.normal(size=P)

    t0 = time.time()
    projector = AffineConstraintProjector(A, B_list, method=method, **kwargs)
    setup_time = time.time() - t0

    t_start = time.time()
    n_done = 0
    for k in range(n_calls):
        projector.project([u.copy() for u in u_list], m.copy(), iteration=k)
        n_done += 1
        if time.time() - t_start > time_budget_s:
            break
    per_call = (time.time() - t_start) / n_done
    return dict(
        method=method,
        I=I,
        L=L,
        P=P,
        setup_time=setup_time,
        per_call_time=per_call,
        n_calls_timed=n_done,
    )


def run_projector_backend_benchmark(pb, dirs, logger, n_calls=20):
    logger.info("=== Section 7a: projector backend benchmark (main dataset) ===")
    rows = []
    for method in ["spsolve", "cached_splu", "smw", "smw_cg"]:
        row = _benchmark_projector(pb.A, pb.B_list, method, n_calls=n_calls)
        rows.append(row)
        logger.info(
            f"  {method}: setup={row['setup_time']:.4f}s, per_call={row['per_call_time']*1000:.3f}ms"
        )
    df = pd.DataFrame(rows)
    df.to_csv(
        os.path.join(dirs["results"], "section7a_projector_backends_main.csv"),
        index=False,
    )
    return df


def run_projector_backend_sweep(
    dirs,
    logger,
    sweep_root=SWEEP_ROOT,
    i_values=(2, 4, 8, 16, 32),
    delta_values=(10, 20, 40),
    n_calls=5,
    time_budget_s=8.0,
    naive_setup_timeout_s=20.0,
    naive_size_cutoff=4000,
):
    """Benchmark the four `AffineConstraintProjector` backends across the
    source-count sweep (I) and the mesh-refinement sweep (delta).

    The "spsolve"/"cached_splu" baselines factor or re-factor the full
    `I*L x I*L` matrix `E E*`, whose fill grows with `O(I^2 L)` and whose
    factorization cost the report itself declines to bound a priori
    ("ordering-dependent... will be measured rather than bounded", Sec. 5.3).
    We measured this directly: at I=32 (L=223), just *factoring* `E E*` once
    for "cached_splu" already takes ~28s on this machine, before a single
    projection is even timed. To keep the sweep bounded we (a) skip these two
    baselines once `I*L` exceeds `naive_size_cutoff`, logging why, and (b)
    wrap every attempt in a hard wall-clock timeout as a safety net in case
    the cutoff underestimates the cost for a given configuration.
    """
    logger.info("=== Section 7b: projector backend cost vs number of sources I ===")
    rows = []
    for n in i_values:
        path = os.path.join(sweep_root, f"I{n}")
        if not os.path.isdir(path):
            logger.warning(f"Sweep dataset {path} not found, skipping I={n}")
            continue
        A, B_list, C, d_list, m = load_experiment_data(path)
        L = A.shape[0]
        methods = ["spsolve", "cached_splu", "smw", "smw_cg"]
        if n * L > naive_size_cutoff:
            logger.info(
                f"  I={n}: I*L={n*L} > {naive_size_cutoff}, skipping spsolve/cached_splu "
                f"(their O(I^2 L)-fill E E* factorization becomes intractable here, "
                f"exactly the failure mode Sec. 5.3 warns about)."
            )
            methods = ["smw", "smw_cg"]
        for method in methods:
            try:
                row = _with_hard_timeout(
                    naive_setup_timeout_s,
                    _benchmark_projector,
                    A,
                    B_list,
                    method,
                    n_calls=n_calls,
                    time_budget_s=time_budget_s,
                )
            except Exception as exc:
                logger.warning(f"  I={n}, method={method} failed/timed out: {exc}")
                continue
            rows.append(row)
            logger.info(
                f"  I={n}, {method}: setup={row['setup_time']:.4f}s, per_call={row['per_call_time']*1000:.3f}ms"
            )
    df_i = pd.DataFrame(rows)
    df_i.to_csv(
        os.path.join(dirs["results"], "section7b_projector_backends_vs_I.csv"),
        index=False,
    )

    logger.info("=== Section 7c: projector backend cost vs mesh refinement (delta) ===")
    rows = []
    for d in delta_values:
        path = os.path.join(sweep_root, f"delta{d}")
        if not os.path.isdir(path):
            logger.warning(f"Sweep dataset {path} not found, skipping delta={d}")
            continue
        A, B_list, C, d_list, m = load_experiment_data(path)
        I_here, L = len(B_list), A.shape[0]
        # smw's dense N_i = A^-1 B_i blows up memory-wise for the largest
        # mesh (delta=40, P=6720); use the matrix-free smw_cg there instead.
        methods = (
            ["spsolve", "cached_splu", "smw", "smw_cg"]
            if d <= 20
            else ["spsolve", "cached_splu", "smw_cg"]
        )
        if I_here * L > naive_size_cutoff:
            logger.info(
                f"  delta={d}: I*L={I_here*L} > {naive_size_cutoff}, skipping spsolve/cached_splu."
            )
            methods = [m_ for m_ in methods if m_ not in ("spsolve", "cached_splu")]
        for method in methods:
            try:
                row = _with_hard_timeout(
                    naive_setup_timeout_s,
                    _benchmark_projector,
                    A,
                    B_list,
                    method,
                    n_calls=n_calls,
                    time_budget_s=time_budget_s,
                )
            except Exception as exc:
                logger.warning(f"  delta={d}, method={method} failed/timed out: {exc}")
                continue
            row["delta"] = d
            rows.append(row)
            logger.info(
                f"  delta={d}, {method}: setup={row['setup_time']:.4f}s, per_call={row['per_call_time']*1000:.3f}ms"
            )
    df_delta = pd.DataFrame(rows)
    df_delta.to_csv(
        os.path.join(dirs["results"], "section7c_projector_backends_vs_delta.csv"),
        index=False,
    )

    if len(df_i) > 0:
        fig, ax = plt.subplots(figsize=(7, 5))
        for method, group in df_i.groupby("method"):
            ax.plot(group["I"], group["per_call_time"] * 1000, marker="o", label=method)
        ax.set_xlabel("Number of sources I (L=223, P=394 fixed)")
        ax.set_ylabel("Per-call projection time (ms)")
        ax.set_yscale("log")
        ax.set_xscale("log", base=2)
        ax.legend()
        ax.set_title(
            "Projector backend cost vs. number of sources (Sec. 5.9, sweep (a))"
        )
        plt.tight_layout()
        plt.savefig(os.path.join(dirs["visuals"], "section7b_projector_vs_I.pdf"))
        plt.close()

    if len(df_delta) > 0:
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        for method, group in df_delta.groupby("method"):
            ax[0].plot(
                group["L"], group["per_call_time"] * 1000, marker="o", label=method
            )
        ax[0].set_xlabel("Field dimension L")
        ax[0].set_ylabel("Per-call projection time (ms)")
        ax[0].set_yscale("log")
        ax[0].set_xscale("log")
        ax[0].legend()
        ax[0].set_title("Projector cost vs. mesh refinement")
        for method, group in df_delta.groupby("method"):
            ax[1].plot(group["L"], group["setup_time"], marker="o", label=method)
        ax[1].set_xlabel("Field dimension L")
        ax[1].set_ylabel("One-time setup time (s)")
        ax[1].set_yscale("log")
        ax[1].set_xscale("log")
        ax[1].legend()
        ax[1].set_title("Setup cost vs. mesh refinement")
        plt.tight_layout()
        plt.savefig(os.path.join(dirs["visuals"], "section7c_projector_vs_delta.pdf"))
        plt.close()

    return df_i, df_delta


def run_mesh_robustness_comparison(
    dirs,
    logger,
    mu=1e-6,
    delta_values=(10, 20, 40),
    max_iterations=3000,
    sweep_root=SWEEP_ROOT,
):
    """Run Algorithm 3 and Algorithm 5 at their theory-derived step sizes
    across mesh refinements, and report objective decrease after a *fixed*
    iteration budget -- the direct test of Sec. 5.1's mesh-robustness claim:
    Algorithm 3's step size shrinks as the mesh degrades (via ||A||-driven
    conditioning, Eq. 33/57), while Algorithm 5's does not (Eq. 46).
    """
    logger.info(
        "=== Section 4b: mesh-refinement robustness of tau/sigma (Alg. 3 vs 5) ==="
    )
    rows = []
    for d in delta_values:
        path = os.path.join(sweep_root, f"delta{d}")
        if not os.path.isdir(path):
            logger.warning(f"Sweep dataset {path} not found, skipping delta={d}")
            continue
        pb = load_problem(path)
        f = objective_data_fidelity(pb)
        x0 = np.zeros(pb.I * pb.L + pb.P, dtype=complex)

        A_lu_normA = power_iteration_operator_norm(
            lambda v: pb.A @ v, lambda w: pb.A_star @ w, dim=pb.L
        )
        normC = power_iteration_operator_norm(
            lambda v: pb.C @ v, lambda w: pb.C_star @ w, dim=pb.L
        )
        # sigma_min(A) via inverse power iteration (matrix-free: only A^-1
        # matvecs through a single sparse LU), needed to check whether it is
        # ||A|| or its conditioning that degrades under mesh refinement
        # (Eq. 57 is stated in terms of sigma_min(A), not ||A|| alone).
        A_lu = sp.linalg.splu(pb.A.tocsc())
        norm_A_inv = power_iteration_operator_norm(
            lambda v: A_lu.solve(v),
            lambda w: A_lu.solve(w, trans="H"),
            dim=pb.L,
        )
        sigma_min_A = 1.0 / norm_A_inv
        cond_A = A_lu_normA / sigma_min_A
        l3 = l_operator_norm_algorithm3(pb, G=None)
        k5 = k_operator_norm_algorithm5(pb, G=None)

        tau3 = sigma3 = 0.9 / l3
        algo3 = ChambollePock(
            exp_name="part45",
            algo_plot_name=f"Alg3-delta{d}",
            f=f,
            A=pb.A,
            B=pb.B_list,
            C=pb.C,
            G=None,
            d=pb.d_list,
            I=pb.I,
            L=pb.L,
            P=pb.P,
            tau=tau3,
            sigma_pde=sigma3,
            prox_dual_reg=None,
        )
        x3 = algo3.run(x0=x0, max_iterations=max_iterations)
        mse3, mae3 = mse_mae(x3, pb.m, pb.P)

        tau5 = sigma5 = 0.9 / k5
        # matrix-free CG projector to stay memory-safe at the finest mesh.
        projector = AffineConstraintProjector(
            pb.A, pb.B_list, method="smw_cg", cg_gamma=0.7
        )
        algo5 = ProjectedChambollePock(
            exp_name="part45",
            algo_plot_name=f"Alg5-delta{d}",
            f=f,
            C=pb.C,
            d=pb.d_list,
            I=pb.I,
            L=pb.L,
            P=pb.P,
            tau=tau5,
            sigma_dat=sigma5,
            sigma_reg=sigma5,
            projector=projector,
            reg_mode="tikhonov",
            mu=mu,
        )
        x5 = algo5.run(x0=x0, max_iterations=max_iterations)
        mse5, mae5 = mse_mae(x5, pb.m, pb.P)

        rows.append(
            dict(
                delta=d,
                L=pb.L,
                P=pb.P,
                normA=A_lu_normA,
                normC=normC,
                sigma_min_A=sigma_min_A,
                cond_A=cond_A,
                l3=l3,
                k5=k5,
                tau3=tau3,
                tau5=tau5,
                iters3=algo3.iteration,
                iters5=algo5.iteration,
                objective3=float(algo3.f_values[-1]),
                objective5=float(algo5.f_values[-1]),
                mse3=mse3,
                mse5=mse5,
            )
        )
        logger.info(
            f"  delta={d}: L={pb.L}, ||A||={A_lu_normA:.3f}, ||C||={normC:.3f}, "
            f"sigma_min(A)={sigma_min_A:.5f}, cond(A)={cond_A:.2f}, "
            f"tau3={tau3:.5f}, tau5={tau5:.5f} ({tau5/tau3:.2f}x larger), "
            f"Alg3 obj={algo3.f_values[-1]:.4g}, Alg5 obj={algo5.f_values[-1]:.4g}"
        )
    df = pd.DataFrame(rows)
    df.to_csv(
        os.path.join(dirs["results"], "section4b_mesh_robustness.csv"), index=False
    )
    return df


# ===========================================================================
# Section 8: noise robustness (extends Table 3 of the internship report)
# ===========================================================================


def run_noise_robustness(
    pb,
    dirs,
    logger,
    noise_levels=(0.0, 1e-2, 5e-2, 1e-1),
    samples=10,
    mu=1e-6,
    lambda_tv=1e-2,
    max_iterations=5000,
    seed=42,
):
    logger.info("=== Section 8: noise robustness across algorithms ===")
    rng = np.random.default_rng(seed)
    lambd, mu1 = 1e-5, 1e-7
    l3_tik = l_operator_norm_algorithm3(pb, G=None)
    l3_tv = l_operator_norm_algorithm3(pb, G=pb.G)
    k5_tik = k_operator_norm_algorithm5(pb, G=None)
    k5_tv = k_operator_norm_algorithm5(pb, G=pb.G)
    x0 = np.zeros(pb.I * pb.L + pb.P, dtype=complex)

    rows = []
    for sigma_noise in noise_levels:
        for sample in range(samples):
            d_list_noisy = [
                d_i
                + sigma_noise
                * (rng.normal(size=d_i.shape) + 1j * rng.normal(size=d_i.shape))
                for d_i in pb.d_list
            ]
            d_noisy = np.concatenate(d_list_noisy, axis=0)
            pb_noisy = ProblemData(
                A=pb.A,
                B_list=pb.B_list,
                C=pb.C,
                d_list=d_list_noisy,
                m=pb.m,
                I=pb.I,
                J=pb.J,
                L=pb.L,
                P=pb.P,
                A_star=pb.A_star,
                C_star=pb.C_star,
                D=pb.D,
                D_star=pb.D_star,
                E=pb.E,
                E_star=pb.E_star,
                d=d_noisy,
                G=pb.G,
            )

            closed = get_closed_form_solution_J_1(pb_noisy, lambd, mu1)
            J1 = get_J_1(pb_noisy, lambd, mu1)
            algo1 = ClosedFormSolution(
                exp_name="part45", algo_plot_name="P-ClosedForm", f=J1, solution=closed
            )
            x1 = algo1.run(x0=x0, max_iterations=1)
            mse1, mae1 = mse_mae(x1, pb.m, pb.P)
            rows.append(
                dict(
                    sigma_noise=sigma_noise,
                    sample=sample,
                    algorithm="P-ClosedForm",
                    mse=mse1,
                    mae=mae1,
                )
            )

            J3 = get_J_3(pb_noisy, mu)
            dJ3 = get_dJ_3(pb_noisy, mu)
            K3 = get_K_J_3(pb_noisy, mu)
            algo3 = NesterovAcceleratedGradientDescent(
                exp_name="part45", algo_plot_name="C-NAGD", f=J3, df=dJ3, K=K3
            )
            m3 = algo3.run(x0=x0[-pb.P :], max_iterations=max_iterations)
            mse3, mae3 = mse_mae(m3, pb.m, pb.P)
            rows.append(
                dict(
                    sigma_noise=sigma_noise,
                    sample=sample,
                    algorithm="C-NAGD",
                    mse=mse3,
                    mae=mae3,
                )
            )

            f = objective_data_fidelity(pb_noisy)

            tau3 = sigma3v = 0.9 / l3_tik
            algo_a3 = ChambollePock(
                exp_name="part45",
                algo_plot_name="Alg3-Tikhonov",
                f=f,
                A=pb.A,
                B=pb.B_list,
                C=pb.C,
                G=sp.eye(pb.P, format="csr"),
                d=d_list_noisy,
                I=pb.I,
                L=pb.L,
                P=pb.P,
                tau=tau3,
                sigma_pde=sigma3v,
                prox_dual_reg=make_tikhonov_dual_prox(mu),
            )
            x_a3 = algo_a3.run(x0=x0, max_iterations=max_iterations)
            mse_a3, mae_a3 = mse_mae(x_a3, pb.m, pb.P)
            rows.append(
                dict(
                    sigma_noise=sigma_noise,
                    sample=sample,
                    algorithm="Alg3-Tikhonov",
                    mse=mse_a3,
                    mae=mae_a3,
                )
            )

            tau3tv = sigma3tv = 0.9 / l3_tv
            algo_a3tv = ChambollePock(
                exp_name="part45",
                algo_plot_name="Alg3-TV",
                f=f,
                A=pb.A,
                B=pb.B_list,
                C=pb.C,
                G=pb.G,
                d=d_list_noisy,
                I=pb.I,
                L=pb.L,
                P=pb.P,
                tau=tau3tv,
                sigma_pde=sigma3tv,
                prox_dual_reg=make_tv_dual_prox(lambda_tv),
            )
            x_a3tv = algo_a3tv.run(x0=x0, max_iterations=max_iterations)
            mse_a3tv, mae_a3tv = mse_mae(x_a3tv, pb.m, pb.P)
            rows.append(
                dict(
                    sigma_noise=sigma_noise,
                    sample=sample,
                    algorithm="Alg3-TV",
                    mse=mse_a3tv,
                    mae=mae_a3tv,
                )
            )

            tau5 = sigma5v = 0.9 / k5_tik
            projector = AffineConstraintProjector(pb.A, pb.B_list, method="smw")
            algo_a5 = ProjectedChambollePock(
                exp_name="part45",
                algo_plot_name="Alg5-Tikhonov",
                f=f,
                C=pb.C,
                d=d_list_noisy,
                I=pb.I,
                L=pb.L,
                P=pb.P,
                tau=tau5,
                sigma_dat=sigma5v,
                sigma_reg=sigma5v,
                projector=projector,
                reg_mode="tikhonov",
                mu=mu,
            )
            x_a5 = algo_a5.run(x0=x0, max_iterations=max_iterations)
            mse_a5, mae_a5 = mse_mae(x_a5, pb.m, pb.P)
            rows.append(
                dict(
                    sigma_noise=sigma_noise,
                    sample=sample,
                    algorithm="Alg5-Tikhonov",
                    mse=mse_a5,
                    mae=mae_a5,
                )
            )

            tau5tv = sigma5tv = 0.9 / k5_tv
            projector_tv = AffineConstraintProjector(pb.A, pb.B_list, method="smw")
            algo_a5tv = ProjectedChambollePock(
                exp_name="part45",
                algo_plot_name="Alg5-TV",
                f=f,
                C=pb.C,
                d=d_list_noisy,
                I=pb.I,
                L=pb.L,
                P=pb.P,
                tau=tau5tv,
                sigma_dat=sigma5tv,
                sigma_reg=sigma5tv,
                projector=projector_tv,
                reg_mode="tv",
                G=pb.G,
                lambda_tv=lambda_tv,
            )
            x_a5tv = algo_a5tv.run(x0=x0, max_iterations=max_iterations)
            mse_a5tv, mae_a5tv = mse_mae(x_a5tv, pb.m, pb.P)
            rows.append(
                dict(
                    sigma_noise=sigma_noise,
                    sample=sample,
                    algorithm="Alg5-TV",
                    mse=mse_a5tv,
                    mae=mae_a5tv,
                )
            )

        logger.info(f"  noise sigma={sigma_noise}: completed {samples} samples")

    df = pd.DataFrame(rows)
    df.to_csv(
        os.path.join(dirs["results"], "section8_noise_robustness_raw.csv"), index=False
    )
    summary = (
        df.groupby(["sigma_noise", "algorithm"])
        .agg(
            mse_mean=("mse", "mean"),
            mse_std=("mse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
        )
        .reset_index()
    )
    summary.to_csv(
        os.path.join(dirs["results"], "section8_noise_robustness_summary.csv"),
        index=False,
    )
    logger.info("Noise robustness summary:\n" + summary.to_string(index=False))

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    for algorithm, group in summary.groupby("algorithm"):
        axs[0].errorbar(
            group["sigma_noise"],
            group["mse_mean"],
            yerr=group["mse_std"],
            marker="o",
            label=algorithm,
            capsize=3,
        )
        axs[1].errorbar(
            group["sigma_noise"],
            group["mae_mean"],
            yerr=group["mae_std"],
            marker="o",
            label=algorithm,
            capsize=3,
        )
    for ax, ylabel in zip(axs, ["MSE", "MAE"]):
        ax.set_xlabel("Noise level (sigma)")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(dirs["visuals"], "section8_noise_robustness.pdf"))
    plt.close()
    return df, summary


# ===========================================================================
# Orchestration
# ===========================================================================


def make_run_dirs(exp_path, exp_name):
    exp_dir, visuals, results = make_dirs(os.path.join(exp_path, exp_name))
    return {"exp": exp_dir, "visuals": visuals, "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="data")
    parser.add_argument("--exp-path", default="runs")
    parser.add_argument("--exp-name", default="part4_5_comparison")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="drastically reduce iteration/sample budgets for a fast smoke run",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dirs = make_run_dirs(args.exp_path, args.exp_name)
    logger = setup_logger(
        name="iwp",
        log_file=os.path.join(dirs["exp"], f"{args.exp_name}.log"),
        level="INFO",
        log_to_console=True,
    )
    set_seed(args.seed)

    pb = load_problem(args.data_path)
    logger.info(f"Loaded problem: I={pb.I}, J={pb.J}, L={pb.L}, P={pb.P}")

    if args.quick:
        iters = dict(
            cnagd=500,
            fista=1000,
            alg35=1000,
            dist=1000,
            stepsize=300,
            block_sigma=300,
            tv=1000,
            exact_inexact=1000,
            mesh=300,
            noise=500,
            noise_samples=2,
        )
    else:
        iters = dict(
            cnagd=5000,
            fista=30000,
            alg35=30000,
            dist=10000,
            stepsize=3000,
            block_sigma=3000,
            tv=10000,
            exact_inexact=8000,
            mesh=3000,
            noise=3000,
            noise_samples=6,
        )

    summary = {}
    _, summary["baselines"] = run_baselines(
        pb,
        dirs,
        logger,
        max_iter_cnagd=iters["cnagd"],
        max_iter_fista=iters["fista"],
    )
    _, summary["algorithm3_5"] = run_algorithm3_and_5(
        pb, dirs, logger, max_iterations=iters["alg35"]
    )
    _, summary["distributed"] = run_distributed_comparison(
        pb, dirs, logger, max_iterations=iters["dist"]
    )
    summary["stepsize"], _ = run_step_size_sensitivity(
        pb, dirs, logger, max_iterations=iters["stepsize"]
    )
    summary["block_sigma"], _, summary["block_norms"] = run_block_sigma_comparison(
        pb, dirs, logger, max_iterations=iters["block_sigma"]
    )
    summary["mesh_robustness"] = run_mesh_robustness_comparison(
        dirs, logger, max_iterations=iters["mesh"]
    )
    _, summary["tv_vs_tikhonov"] = run_tv_vs_tikhonov(
        pb, dirs, logger, max_iterations=iters["tv"]
    )
    _, summary["exact_vs_inexact"] = run_exact_vs_inexact_projection(
        pb, dirs, logger, max_iterations=iters["exact_inexact"]
    )
    summary["projector_main"] = run_projector_backend_benchmark(pb, dirs, logger)
    summary["projector_I"], summary["projector_delta"] = run_projector_backend_sweep(
        dirs, logger
    )
    _, summary["noise"] = run_noise_robustness(
        pb,
        dirs,
        logger,
        samples=iters["noise_samples"],
        max_iterations=iters["noise"],
    )

    for name, df in summary.items():
        if isinstance(df, pd.DataFrame):
            logger.info(f"--- {name} ---\n{df.to_string(index=False)}")

    logger.info(f"All results saved under {dirs['exp']}")


if __name__ == "__main__":
    main()
