#!/usr/bin/env python
"""Acceptance checks for the acceleration and line-search flags of Sec. 4.8.

Four things are checked, in the order they can break:

1. The block-weighted affine projection agrees with a dense reference
   `z - T E^*(E T E^*)^-1 E z` for several step-size ratios, on both SMW
   backends. Everything else is built on this, so it is checked first and
   against a formula rather than against itself.
2. `accelerate="none", linesearch=False` reproduces the pre-flag iteration
   *bit for bit*, for every regularizer mode. The old `step` is replayed
   verbatim from a copy kept in this file, so this is a real comparison and
   not a tautology.
3. `weight_ratio=1.0` takes the untouched code path in the projector.
4. The documented error paths raise, rather than silently running something
   that has no convergence theory behind it.

Run:
    pixi run python scripts/validate_acceleration_flags.py
Exits non-zero if any check fails.
"""

import os
import sys

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

from iwp.algorithms.algorithms import (  # noqa: E402
    AffineConstraintProjector,
    ChambollePock,
    ProjectedChambollePock,
)
from iwp.experiments.comparison import (  # noqa: E402
    k_operator_norm_algorithm5,
    load_problem,
)

FAILURES = []


def check(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not passed:
        FAILURES.append(name)


class LegacyProjectedChambollePock(ProjectedChambollePock):
    """The `step` this class had before the flags landed, kept verbatim so
    check 2 compares against the real thing rather than a paraphrase."""

    def step(self, x):
        u, m = self._split(x)
        if self.x_bar is None:
            u_bar, m_bar = u, m
        else:
            u_bar, m_bar = self._split(self.x_bar)

        v_dat_new = [
            self.data_dual_prox[i](
                self.v_dat[i] + self.sigma_dat * (self.C @ u_bar[i]), self.sigma_dat
            )
            for i in range(self.I)
        ]

        if self.reg_mode == "tv":
            v_reg_new = self.prox_reg_dual(
                self.v_reg + self.sigma_reg * (self.G @ m_bar), self.sigma_reg
            )
        elif self.reg_mode == "tikhonov":
            v_reg_new = self.prox_reg_dual(
                self.v_reg + self.sigma_reg * m_bar, self.sigma_reg
            )
        else:
            v_reg_new = self.v_reg

        z_u = [u[i] - self.tau * (self.C_star @ v_dat_new[i]) for i in range(self.I)]
        if self.reg_mode == "tv":
            z_m = m - self.tau * (self.G_star @ v_reg_new)
        elif self.reg_mode == "tikhonov":
            z_m = m - self.tau * v_reg_new
        else:
            z_m = m.copy()

        u_new, m_new = self.projector.project(z_u, z_m, iteration=self.iteration)

        x_new = np.empty_like(x)
        for i in range(self.I):
            x_new[i * self.L : (i + 1) * self.L] = u_new[i]
        x_new[self.I * self.L : self.I * self.L + self.P] = m_new

        if self.x_bar is None:
            self.x_bar = np.empty_like(x)
        for i in range(self.I):
            self.x_bar[i * self.L : (i + 1) * self.L] = 2 * u_new[i] - u[i]
        self.x_bar[self.I * self.L : self.I * self.L + self.P] = 2 * m_new - m

        self.v_dat = v_dat_new
        self.v_reg = v_reg_new
        self.current_residual = self.projector.feasibility_residual_norm(u_new, m_new)
        self.current_step_norm = np.linalg.norm(x_new - x)
        return x_new


def data_objective(pb):
    def f(x):
        total = 0.0
        for i in range(pb.I):
            r = pb.C @ x[i * pb.L : (i + 1) * pb.L] - pb.d_list[i]
            total += 0.5 * np.vdot(r, r).real
        return float(total)

    return f


def check_weighted_projection(pb):
    rng = np.random.default_rng(0)
    zu = [rng.standard_normal(pb.L) + 1j * rng.standard_normal(pb.L)
          for _ in range(pb.I)]
    zm = rng.standard_normal(pb.P) + 1j * rng.standard_normal(pb.P)
    z = np.concatenate(list(zu) + [zm])
    E = pb.E.toarray()

    for method, kwargs, tol in (
        ("smw", {}, 1e-12),
        ("smw_cg", dict(cg_eta0=1e-12, cg_min_tol=1e-14, cg_maxiter=800), 1e-10),
    ):
        proj = AffineConstraintProjector(pb.A, pb.B_list, method=method, **kwargs)
        u0, m0 = proj.project([q.copy() for q in zu], zm.copy())
        u1, m1 = proj.project([q.copy() for q in zu], zm.copy(), weight_ratio=1.0)
        check(
            f"projector[{method}]: weight_ratio=1.0 is the untouched path",
            all(np.array_equal(a, b) for a, b in zip(u0, u1)) and np.array_equal(m0, m1),
        )
        worst = 0.0
        for c in (0.25, 0.5, 2.0, 17.0, 137.0):
            un, mn = proj.project([q.copy() for q in zu], zm.copy(), weight_ratio=c)
            T = np.diag(np.concatenate(
                [np.full(pb.I * pb.L, c), np.full(pb.P, 1.0)]))
            x_ref = z - T @ E.conj().T @ np.linalg.solve(E @ T @ E.conj().T, E @ z)
            x = np.concatenate(list(un) + [mn])
            worst = max(worst, np.linalg.norm(x - x_ref) / np.linalg.norm(x_ref))
        check(
            f"projector[{method}]: weighted projection matches the dense formula",
            worst < tol,
            f"worst relative error {worst:.2e}",
        )


def check_bit_for_bit(pb, iterations=400):
    f = data_objective(pb)
    x0 = np.zeros(pb.I * pb.L + pb.P, dtype=complex)
    for reg_mode, kwargs in (
        ("tikhonov", dict(mu=1e-6)),
        ("tv", dict(G=pb.G, lambda_tv=1e-3)),
        ("none", dict()),
    ):
        k5 = k_operator_norm_algorithm5(pb, G=kwargs.get("G"))
        tau = sigma = 0.9 / k5
        outs = []
        for cls in (LegacyProjectedChambollePock, ProjectedChambollePock):
            projector = AffineConstraintProjector(pb.A, pb.B_list, method="smw")
            algo = cls(
                "validate", "validate", f, pb.C, pb.d_list, pb.I, pb.L, pb.P,
                tau, sigma, sigma, projector, reg_mode=reg_mode, **kwargs,
            )
            outs.append(algo.run(x0=x0, max_iterations=iterations))
        check(
            f"Alg5 defaults reproduce the pre-flag iteration bit for bit "
            f"(reg_mode={reg_mode!r})",
            np.array_equal(outs[0], outs[1]),
            f"max |diff| = {np.max(np.abs(outs[0] - outs[1])):.3e} "
            f"after {iterations} iterations",
        )


def check_error_paths(pb):
    f = data_objective(pb)
    projector = AffineConstraintProjector(pb.A, pb.B_list, method="smw")
    common = dict(
        exp_name="validate", algo_plot_name="validate", f=f, C=pb.C, d=pb.d_list,
        I=pb.I, L=pb.L, P=pb.P, tau=0.4, sigma_dat=0.4, sigma_reg=0.4,
        projector=projector,
    )

    def raises(fn, exc=ValueError):
        try:
            fn()
        except exc:
            return True
        except Exception:
            return False
        return False

    check(
        "subspace + reg_mode='tv' without gamma raises",
        raises(lambda: ProjectedChambollePock(
            **common, reg_mode="tv", G=pb.G, lambda_tv=1e-3, accelerate="subspace")),
    )
    check(
        "subspace + reg_mode='none' raises",
        raises(lambda: ProjectedChambollePock(
            **common, reg_mode="none", accelerate="subspace")),
    )
    check(
        "subspace + reg_mode='tikhonov' with gamma != mu raises",
        raises(lambda: ProjectedChambollePock(
            **common, reg_mode="tikhonov", mu=1e-3, accelerate="subspace",
            gamma=1e-2)),
    )
    check(
        "linesearch + dual_data raises",
        raises(lambda: ProjectedChambollePock(
            **common, reg_mode="tikhonov", mu=1e-3, accelerate="dual_data",
            linesearch=True)),
    )
    check(
        "unknown accelerate mode raises",
        raises(lambda: ProjectedChambollePock(
            **common, reg_mode="tikhonov", mu=1e-3, accelerate="full")),
    )
    check(
        "out-of-range backtracking parameter raises",
        raises(lambda: ProjectedChambollePock(
            **common, reg_mode="tikhonov", mu=1e-3, linesearch=True, mu_ls=1.4)),
    )
    check(
        "Algorithm 3 refuses acceleration and says why",
        raises(lambda: ChambollePock(
            exp_name="validate", algo_plot_name="validate", f=f, A=pb.A, B=pb.B_list,
            C=pb.C, G=None, d=pb.d_list, I=pb.I, L=pb.L, P=pb.P, tau=0.1, sigma=0.1,
            accelerate="subspace"), NotImplementedError),
    )
    check(
        "weighted projection is refused by the non-SMW backends",
        raises(lambda: AffineConstraintProjector(
            pb.A, pb.B_list, method="cached_splu").project(
                [np.zeros(pb.L, dtype=complex) for _ in range(pb.I)],
                np.zeros(pb.P, dtype=complex), weight_ratio=2.0),
            NotImplementedError),
    )


def main():
    pb = load_problem("data")
    print(f"Problem: I={pb.I}, J={pb.J}, L={pb.L}, P={pb.P}\n")
    check_weighted_projection(pb)
    print()
    check_bit_for_bit(pb)
    print()
    check_error_paths(pb)
    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {', '.join(FAILURES)}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
