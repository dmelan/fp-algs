import abc
import os
import time
import tracemalloc

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp

from .metrics import mae, mse

# ---------------------------------------------------------------------------
# Dual proximal operators shared by the Chambolle-Pock family (Sec. 4.7, 5.2).
#
# All of them share the signature `prox(v, sigma) -> v_proj`, even when the
# operator does not actually depend on `sigma` (the projection onto a fixed
# norm ball does not, since it is the conjugate of an indicator function, cf.
# Eq. (27), (45)): this lets `ChambollePock`/`ProjectedChambollePock` treat a
# Total Variation and a Tikhonov regularizer interchangeably.
# ---------------------------------------------------------------------------


def group_l2inf_ball_projection(v, radius, group_size=1):
    """Euclidean projection onto the group l2,inf ball of the given radius,
    i.e. the polar ball of the group l2,1 norm used to dualize a Total
    Variation-type regularizer (Eq. (27) for the centralized/distributed
    Chambolle-Pock of Sec. 4.7, Eq. (45) for the projected scheme of
    Sec. 5.2). `group_size=1` (the default) recovers plain componentwise
    l_infty clipping, which is what the group l2,1 norm reduces to for a P0
    (piecewise-constant) contrast basis with singleton jump groups; see the
    remark in Sec. 5.2.
    """
    v = np.asarray(v)
    if v.size == 0:
        return v
    if group_size == 1:
        mag = np.abs(v)
        scale = np.minimum(1.0, radius / np.maximum(mag, 1e-300))
        return v * scale
    n_groups = v.shape[0] // group_size
    v_reshaped = v.reshape(n_groups, group_size)
    norms = np.linalg.norm(v_reshaped, axis=1, keepdims=True)
    scale = np.minimum(1.0, radius / np.maximum(norms, 1e-300))
    return (v_reshaped * scale).reshape(-1)


def make_tv_dual_prox(lambda_tv, group_size=1):
    """`prox_{sigma g*}` for `g = lambda_tv * ||.||_{2,1}` (Eq. (45)):
    independent of `sigma` since it is an indicator's conjugate."""

    def prox(v, sigma=None):
        return group_l2inf_ball_projection(v, lambda_tv, group_size=group_size)

    return prox


def make_tikhonov_dual_prox(mu):
    """`prox_{sigma g*}(v) = mu * v / (mu + sigma)` for `g = (mu/2)||.||_2^2`
    (remark following Eq. (46)): used to dualize a Tikhonov regularizer on m
    instead of Total Variation, e.g. inside `ChambollePock` or
    `ProjectedChambollePock`'s regularizer block."""

    def prox(v, sigma):
        return (mu / (mu + sigma)) * v

    return prox


def make_data_dual_prox(d_i):
    """`prox_{sigma g*_dat}(v) = (v - sigma * d_i) / (1 + sigma)` (Eq. (44)):
    the dual of the quadratic data-fidelity term `(1/2)||. - d_i||_2^2`, used
    to dualize the data term in `ProjectedChambollePock` (Algorithm 5,
    Sec. 5.5) instead of proxing it as in `ChambollePock` (Algorithm 3)."""

    def prox(v, sigma):
        return (v - sigma * d_i) / (1.0 + sigma)

    return prox


class FixedPointAlgorithm(abc.ABC):
    def __init__(self, exp_name, algo_plot_name, f, logger=None, verbose=False):
        self.exp_name = exp_name
        self.algo_plot_name = algo_plot_name
        self.f = f
        self.x_values = []
        self.f_values = []
        self.iteration = None
        self.max_iterations = None
        self.cv_time = None
        self.memory_used = None
        self.logger = logger
        self.verbose = verbose

    @abc.abstractmethod
    def step(self, x):
        pass

    @abc.abstractmethod
    def is_converged(self, x):
        pass

    def run(self, x0, max_iterations=1000, store_history=True):
        """Run the fixed-point iteration from `x0`.

        Args:
            store_history: keep every iterate in `self.x_values` (the default,
                which every existing caller and `plot_algorithm_convergence`
                relies on). Set False to keep only the last iterate: the
                preallocation costs `(max_iterations+1) * dim * 16` bytes,
                which is 655 MB for a 3000-iteration run at delta=40, and the
                paired proxy/FE studies only ever read the final iterate and
                the objective trace. `self.x_values[-1]` keeps working either
                way; `f_values` is always kept in full (it is one float per
                iteration).
        """
        self.max_iterations = max_iterations
        self.store_history = store_history
        if self.logger:
            self.logger.info(
                f"Started {self.algo_plot_name} for a maximum of {self.max_iterations} iterations."
            )
        # Preallocate arrays to not count them in memory usage
        self.x_values = np.empty(
            ((max_iterations + 1) if store_history else 1,) + x0.shape, dtype=x0.dtype
        )
        self.f_values = np.empty(max_iterations + 1, dtype=float)
        # Start measuring time and memory
        tracemalloc.start()
        t0 = time.time()
        x = x0
        self.x_values[0] = x0
        self.f_values[0] = self.f(x0)
        self.iteration = 0
        while not self.is_converged(x) and self.iteration < self.max_iterations:
            x = self.step(x)
            self.iteration += 1
            self.x_values[self.iteration if store_history else 0] = x
            self.f_values[self.iteration] = self.f(x)
            if self.logger:
                msg = f"Iteration {self.iteration}: f(x) = {self.f_values[self.iteration]:.6f}, time = {time.time() - t0:.3f}s"
                (self.logger.info if self.verbose else self.logger.debug)(msg)
        # Special case for the closed-form algorithm
        if self.iteration == 0:
            x = self.step(x)
            self.iteration += 1
            self.x_values[self.iteration if store_history else 0] = x
            self.f_values[self.iteration] = self.f(x)
        self.cv_time = time.time() - t0
        _, peak = tracemalloc.get_traced_memory()
        self.memory_used = peak
        tracemalloc.stop()
        if self.logger:
            msg = (
                f"Converged after {self.iteration} iterations in {self.cv_time:.3f} seconds with {self.memory_used / 1024:.2f} KB memory used."
                if self.iteration < self.max_iterations
                else f"Stopped after {self.max_iterations} iterations in {self.cv_time:.3f} seconds with {self.memory_used / 1024:.2f} KB memory used."
            )
            self.logger.info(msg)
        # Cut arrays to actual size
        if store_history:
            self.x_values = self.x_values[: self.iteration + 1]
        self.f_values = self.f_values[: self.iteration + 1]
        return x

    def plot_algorithm_convergence(
        self, m, visuals_path, add_marker=False, show=False, save=True
    ):
        # MSE/MAE curves need the full iterate history; `run(store_history=False)`
        # keeps only the last iterate, which would silently plot a one-point
        # "convergence" curve. Fail loudly instead.
        if getattr(self, "store_history", True) is False:
            raise RuntimeError(
                f"{self.algo_plot_name}: plot_algorithm_convergence needs the iterate "
                "history, but run(..., store_history=False) discarded it. Re-run with "
                "store_history=True (the default) to plot MSE/MAE per iteration."
            )
        m_pred = (
            self.x_values[:, -m.shape[0] :]
            if self.x_values.ndim == 2
            else self.x_values
        )
        mse_values = mse(m_pred, m)
        mae_values = mae(m_pred, m)
        self.mse_values = mse_values
        self.mae_values = mae_values

        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        for ax, values, label, ylabel in zip(
            axs,
            [mse_values, mae_values, self.f_values],
            ["MSE", "MAE", "Objective function"],
            ["MSE", "MAE", "Objective function"],
        ):
            if add_marker:
                ax.plot(values, label=label, marker="o", markersize=4)
            else:
                ax.plot(values, label=label)
            if self.iteration < self.max_iterations:
                ax.scatter(
                    self.iteration, values[-1], color="red", marker="x", label="Stopped"
                )
            ax.set_xlabel("Iteration")
            ax.set_ylabel(ylabel)
            if ylabel == "Objective function":
                # Log scale only for objective function
                ax.set_yscale("log")
            ax.legend()

        fig.suptitle(
            f"Convergence Plots for {self.algo_plot_name} in {self.cv_time:.3f}s using {self.memory_used / 1024:.2f} KB"
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if show:
            plt.show()
        if save:
            file_name = os.path.join(visuals_path, self.algo_plot_name + ".pdf")
            plt.savefig(file_name)
            if self.logger:
                self.logger.info(f"Saved convergence plots to {file_name}")
        plt.close()


class ClosedFormSolution(FixedPointAlgorithm):
    def __init__(
        self, exp_name, algo_plot_name, f, solution, logger=None, verbose=False
    ):
        super().__init__(exp_name, algo_plot_name, f, logger=logger, verbose=verbose)
        self.solution = solution

    def step(self, x):
        return self.solution()

    def is_converged(self, x, threshold=1e-6):
        return True


class GradientDescent(FixedPointAlgorithm):
    def __init__(
        self, exp_name, algo_plot_name, f, df, K, gamma, logger=None, verbose=False
    ):
        super().__init__(exp_name, algo_plot_name, f, logger=logger, verbose=verbose)
        self.df = df
        self.K = K
        assert gamma < 2.0 / self.K, "gamma must be less than 2/L for convergence"
        self.gamma = gamma
        self.current_gradient = None

    def step(self, x):
        return x - self.gamma * self.current_gradient

    def is_converged(self, x, threshold=1e-6):
        self.current_gradient = self.df(x)
        return np.linalg.norm(self.current_gradient) < threshold


class NesterovAcceleratedGradientDescent(FixedPointAlgorithm):
    def __init__(self, exp_name, algo_plot_name, f, df, K, logger=None, verbose=False):
        super().__init__(exp_name, algo_plot_name, f, logger=logger, verbose=verbose)
        self.df = df
        self.K = K
        self.beta_prev = 1.0
        self.y_prev = None
        self.current_gradient = None

    def step(self, x):
        if self.y_prev is None:
            self.y_prev = x
        y_new = x - (1.0 / self.K) * self.current_gradient
        self.beta = (1 + np.sqrt(1 + 4 * self.beta_prev**2)) / 2
        self.gamma = (self.beta_prev - 1) / self.beta
        self.beta_prev = self.beta
        x_new = y_new + self.gamma * (y_new - self.y_prev)
        self.y_prev = y_new.copy()
        return x_new

    def is_converged(self, x, threshold=1e-6):
        self.current_gradient = self.df(x)
        return np.linalg.norm(self.current_gradient) < threshold


class StronglyConvexNesterovAcceleratedGradientDescent(FixedPointAlgorithm):
    def __init__(
        self, exp_name, algo_plot_name, f, df, K, mu, logger=None, verbose=False
    ):
        super().__init__(exp_name, algo_plot_name, f, logger=logger, verbose=verbose)
        self.df = df
        self.K = K
        self.mu = mu
        ratio = np.sqrt(K / mu)
        self.gamma = -(ratio - 1) / (ratio + 1)
        self.y_prev = None
        self.current_gradient = None

    def step(self, x):
        if self.y_prev is None:
            self.y_prev = x
        y_new = x - (1.0 / self.K) * self.current_gradient
        x_new = y_new + self.gamma * (y_new - self.y_prev)
        self.y_prev = y_new.copy()
        return x_new

    def is_converged(self, x, threshold=1e-6):
        self.current_gradient = self.df(x)
        return np.linalg.norm(self.current_gradient) < threshold


class ForwardBackward(FixedPointAlgorithm):
    def __init__(
        self,
        exp_name,
        algo_plot_name,
        f,
        grad,
        prox,
        gamma,
        lambd,
        logger=None,
        verbose=False,
    ):
        super().__init__(exp_name, algo_plot_name, f, logger=logger, verbose=verbose)
        self.grad = grad
        self.prox = prox
        self.gamma = gamma
        self.lambd = lambd
        self.current_gradient = None

    def step(self, x):
        gamma_n = self.gamma(self.iteration) if callable(self.gamma) else self.gamma
        lambda_n = self.lambd(self.iteration) if callable(self.lambd) else self.lambd
        z = x - gamma_n * self.current_gradient
        y = self.prox(z, gamma_n)
        return x + lambda_n * (y - x)

    def is_converged(self, x, threshold=1e-6):
        self.current_gradient = self.grad(x)
        return np.linalg.norm(self.current_gradient) < threshold


class FISTA(FixedPointAlgorithm):
    def __init__(
        self,
        exp_name,
        algo_plot_name,
        f,
        grad,
        prox,
        K,
        logger=None,
        verbose=False,
    ):
        super().__init__(exp_name, algo_plot_name, f, logger=logger, verbose=verbose)
        self.grad = grad
        self.prox = prox
        self.K = K
        self.beta_prev = 1.0
        self.y_prev = None
        self.current_gradient = None

    def step(self, x):
        if self.y_prev is None:
            self.y_prev = x
        z = x - (1.0 / self.K) * self.current_gradient
        y = self.prox(z, self.K)
        self.beta = (1 + np.sqrt(1 + 4 * self.beta_prev**2)) / 2
        self.gamma = (self.beta_prev - 1) / self.beta
        self.beta_prev = self.beta
        x_new = y + self.gamma * (y - self.y_prev)
        self.y_prev = y.copy()
        return x_new

    def is_converged(self, x, threshold=1e-6):
        self.current_gradient = self.grad(x)
        return np.linalg.norm(self.current_gradient) < threshold


class ChambollePock(FixedPointAlgorithm):
    """Chambolle-Pock primal-dual algorithm (Algorithm 3, Sec. 4.7) for the
    constrained inverse problem: the PDE constraint `A u_i = B_i m` and the
    regularizer on m are dualized (through `v_pde` and `v_reg`), while the
    quadratic data-fidelity term is kept in the primal and evaluated exactly
    through its proximity operator (Sherman-Morrison-Woodbury on `C`,
    Eq. (29)-(30)). This is the complementary instantiation to
    `ProjectedChambollePock` (Algorithm 5, Sec. 5.5), which instead keeps the
    PDE constraint in the primal (via an exact/inexact affine projection) and
    dualizes the data term.

    State layout: x = [u_0, ..., u_{I-1}, m] with length I*L + P. `A` is the
    shared Born-linearized PDE operator, `B[i]` the coupling operator for
    source i, `C` the shared observation operator and `G` the operator used
    to dualize the regularizer on m (the discrete gradient/graph operator for
    Total Variation, or the identity for Tikhonov; pass `G=None` to disable
    regularization entirely). `prox_dual_reg(v, sigma)` is the proximity
    operator of the *conjugate* of the regularizer (e.g. `make_tv_dual_prox`
    for the group l2,inf ball projection of Eq. (27), or
    `make_tikhonov_dual_prox`).

    Note on notation: the report denotes the discrete-gradient/TV-dualization
    operator `D` in Sec. 4.7 (Eq. (26)) but flags a clash with the `D` used
    for the stacked measurement operator in Sec. 2-3 (see the remark opening
    Sec. 5); this implementation follows the Sec. 5 convention and calls it
    `G` throughout to avoid the ambiguity.
    """

    def __init__(
        self,
        exp_name,
        algo_plot_name,
        f,
        A,
        B,
        C,
        G,
        d,
        I,
        L,
        P,
        tau,
        sigma,
        prox_dual_reg=None,
        accelerate="none",
        linesearch=False,
        mu_ls=0.7,
        delta_ls=0.99,
        ls_max_trials=60,
        logger=None,
        verbose=False,
    ):
        super().__init__(exp_name, algo_plot_name, f, logger=logger, verbose=verbose)
        if accelerate != "none":
            raise NotImplementedError(
                "ChambollePock has no admissible acceleration schedule. Algorithm 3 "
                "dualizes the PDE constraint and the regularizer under a *single* "
                "scalar sigma, and the PDE block has g* = 0, which is not strongly "
                "convex; any schedule that grows sigma for the regularizer block "
                "grows it for the PDE block too and breaks tau*sigma*||L||^2 <= 1. "
                "Partial acceleration needs the disjoint block structure that only "
                "ProjectedChambollePock (Algorithm 5) has. Use linesearch=True here, "
                "which is well defined for a scalar metric."
            )
        self.A = A
        self.A_star = A.conj().T
        self.B = B
        self.B_star = [Bi.conj().T for Bi in B]
        self.C = C
        self.C_star = C.conj().T
        self.G = G
        self.G_star = G.conj().T if G is not None else None
        self.d = d
        self.I = I
        self.L = L
        self.P = P
        self.tau = tau
        self.sigma = sigma
        self.prox_dual_reg = prox_dual_reg

        # One-time setup: factor the Woodbury correction matrix (I + tau*C*C^*),
        # shared across all sources since C does not depend on i, cf. Eq. (32).
        # It depends on tau, so the line search refactors it whenever tau
        # moves; that is a J x J solve (J = 50 here) against the I*L-scale
        # work of the rest of the iteration, so it is not a cost worth
        # avoiding.
        self._woodbury_tau = None
        self._refactor_woodbury(tau)

        Q = G.shape[0] if G is not None else 0
        self.v_pde = [np.zeros(L, dtype=complex) for _ in range(I)]
        self.v_reg = np.zeros(Q, dtype=complex)
        self.u_bar = None
        self.m_bar = None
        self.current_residual = None

        self.linesearch = bool(linesearch)
        if not (0.0 < mu_ls < 1.0):
            raise ValueError(f"mu_ls must lie in (0,1), got {mu_ls}")
        if not (0.0 < delta_ls < 1.0):
            raise ValueError(f"delta_ls must lie in (0,1), got {delta_ls}")
        self.mu_ls = mu_ls
        self.delta_ls = delta_ls
        self.ls_max_trials = int(ls_max_trials)
        self._beta_ls = sigma / tau
        self._theta_prev = 1.0
        self.tau_history = []
        self.ls_trials = []
        self.n_A_matvecs = 0
        self.n_C_applies = 0

    def _refactor_woodbury(self, tau):
        if self._woodbury_tau == tau:
            return
        J = self.C.shape[0]
        woodbury_matrix = sp.eye(J, format="csc", dtype=complex) + tau * (
            self.C @ self.C_star
        )
        self.woodbury_lu = sp.linalg.splu(woodbury_matrix.tocsc())
        self._woodbury_tau = tau

    def _prox_data(self, i, v):
        # (C^*C + tau^-1 I)^-1 (C^*d_i + tau^-1 v) via Sherman-Morrison-Woodbury, Eq. (31)-(32)
        rhs = self.C_star @ self.d[i] + v / self.tau
        correction = self.C_star @ self.woodbury_lu.solve(self.C @ rhs)
        return self.tau * (rhs - self.tau * correction)

    def step(self, x):
        return self._step_linesearch(x) if self.linesearch else self._step_plain(x)

    def _grad_m(self, v_pde, v_reg):
        grad_m = sum(self.B_star[i] @ v_pde[i] for i in range(self.I))
        if self.G is not None:
            grad_m = grad_m - self.G_star @ v_reg
        return grad_m

    def _step_linesearch(self, x):
        """Malitsky-Pock Algorithm 4 on the scalar metric. The primal step
        runs once at the previously accepted tau; only the dual update is
        repeated. Unlike Algorithm 5, where a rejected trial costs `I` cheap
        applications of `C`, here it costs `I` applications of `A` and `B`,
        so the useful accounting is in `n_A_matvecs`, not iterations."""
        L, P, I = self.L, self.P, self.I
        m_start = I * L
        u = [x[i * L : (i + 1) * L] for i in range(I)]
        m = x[m_start : m_start + P]

        self._refactor_woodbury(self.tau)
        u_new = [
            self._prox_data(i, u[i] - self.tau * (self.A_star @ self.v_pde[i]))
            for i in range(I)
        ]
        self.n_A_matvecs += I
        self.n_C_applies += 2 * I  # one C and one C^* inside each Woodbury prox
        m_new = m + self.tau * self._grad_m(self.v_pde, self.v_reg)

        tau_prev = self.tau
        scale = np.sqrt(1.0 + self._theta_prev)
        v_pde_new = v_reg_new = None
        theta = 1.0
        accepted = False
        for trial in range(1, self.ls_max_trials + 1):
            theta = scale
            tau = tau_prev * scale
            sigma = self._beta_ls * tau
            u_bar = [u_new[i] + theta * (u_new[i] - u[i]) for i in range(I)]
            m_bar = m_new + theta * (m_new - m)

            v_pde_new = [
                self.v_pde[i] + sigma * (self.A @ u_bar[i] - self.B[i] @ m_bar)
                for i in range(I)
            ]
            self.n_A_matvecs += I
            v_reg_new = (
                self.prox_dual_reg(self.v_reg + sigma * (self.G @ m_bar), sigma)
                if self.G is not None
                else self.v_reg
            )

            dv_pde = [v_pde_new[i] - self.v_pde[i] for i in range(I)]
            dv_reg = (
                v_reg_new - self.v_reg if self.G is not None else np.zeros(0, dtype=complex)
            )
            # L^* dv = ((A^* dv_pde_i)_i, -sum_i B_i^* dv_pde_i + G^* dv_reg)
            l_star_u = [self.A_star @ dv for dv in dv_pde]
            self.n_A_matvecs += I
            l_star_m = -self._grad_m(dv_pde, dv_reg)
            lhs = tau * (
                sum(np.linalg.norm(w) ** 2 for w in l_star_u)
                + np.linalg.norm(l_star_m) ** 2
            )
            dv_sq = sum(np.linalg.norm(dv) ** 2 for dv in dv_pde) + (
                np.linalg.norm(dv_reg) ** 2
            )
            rhs = (self.delta_ls**2) * dv_sq / sigma
            if lhs <= rhs or dv_sq == 0.0:
                accepted = True
                break
            scale = scale * self.mu_ls
        self.ls_trials.append(trial)
        if not accepted and self.logger:
            self.logger.warning(
                f"{self.algo_plot_name}: line search hit its trial cap "
                f"({self.ls_max_trials}) at iteration {self.iteration}; accepting "
                f"tau={tau_prev * theta:.4g} without the descent test."
            )

        self.tau = tau_prev * theta
        self.sigma = self._beta_ls * self.tau
        self._theta_prev = theta
        self.v_pde = v_pde_new
        self.v_reg = v_reg_new
        self.tau_history.append(self.tau)

        self.current_residual = np.sqrt(
            sum(
                np.linalg.norm(self.A @ u_new[i] - self.B[i] @ m_new) ** 2
                for i in range(I)
            )
        )
        self.n_A_matvecs += I
        x_new = np.empty_like(x)
        for i in range(I):
            x_new[i * L : (i + 1) * L] = u_new[i]
        x_new[m_start : m_start + P] = m_new
        return x_new

    def _step_plain(self, x):
        u = [x[i * self.L : (i + 1) * self.L] for i in range(self.I)]
        m_start = self.I * self.L
        m = x[m_start : m_start + self.P]
        self.n_A_matvecs += 3 * self.I
        self.n_C_applies += 2 * self.I
        self.tau_history.append(self.tau)

        if self.u_bar is None:
            self.u_bar = [ui.copy() for ui in u]
            self.m_bar = m.copy()

        # PDE dual update (parallelizable over i)
        for i in range(self.I):
            self.v_pde[i] = self.v_pde[i] + self.sigma * (
                self.A @ self.u_bar[i] - self.B[i] @ self.m_bar
            )
        # Regularizer dual update (TV or Tikhonov, cf. `prox_dual_reg`)
        if self.G is not None:
            self.v_reg = self.prox_dual_reg(
                self.v_reg + self.sigma * (self.G @ self.m_bar), self.sigma
            )

        # Primal update of the u_i's (parallelizable over i)
        u_new = [
            self._prox_data(i, u[i] - self.tau * (self.A_star @ self.v_pde[i]))
            for i in range(self.I)
        ]

        # Primal update of m: plain gradient combination of dual variables
        grad_m = sum(self.B_star[i] @ self.v_pde[i] for i in range(self.I))
        if self.G is not None:
            grad_m = grad_m - self.G_star @ self.v_reg
        m_new = m + self.tau * grad_m

        # Over-relaxation (theta = 1)
        self.u_bar = [2 * u_new[i] - u[i] for i in range(self.I)]
        self.m_bar = 2 * m_new - m

        self.current_residual = np.sqrt(
            sum(
                np.linalg.norm(self.A @ u_new[i] - self.B[i] @ m_new) ** 2
                for i in range(self.I)
            )
        )

        x_new = np.empty_like(x)
        for i in range(self.I):
            x_new[i * self.L : (i + 1) * self.L] = u_new[i]
        x_new[m_start : m_start + self.P] = m_new
        return x_new

    def is_converged(self, x, threshold=1e-6):
        return self.current_residual is not None and self.current_residual < threshold


class AffineConstraintProjector:
    """Projects `x = [u_0, ..., u_{I-1}, m]` onto the affine PDE-feasibility
    set `C = {x : A u_i = B_i m, i = 0, ..., I-1}`, equivalently `E x = 0`
    with `E = [A_I, -B]` (block-diagonal `A_I = I_I kron A` and stacked
    `B = [B_0; ...; B_{I-1}]`, Eq. (47)). Used by `ProjectedChambollePock`
    (Algorithm 5, Sec. 5.5) to evaluate `prox_{tau iota_C}` exactly (or
    inexactly), and reused by the comparison notebook to plug the same
    projection into `FISTA`/`ForwardBackward`'s `prox` argument for a fair
    like-for-like comparison of projection *backends* (Table 2, Sec. 5.8-5.9).

    Four interchangeable backends are offered:

    - "spsolve": forms `E E*` and solves `(E E*) w = E x` from scratch on
      every call, exactly what the original report's FISTA `prox_J_2`
      does (`sp.sparse.linalg.spsolve(E @ E_star, E @ x)`); the "S1" baseline
      of the experiment plan.
    - "cached_splu": same `E E*` system, but its sparse LU is factored once
      at construction and reused for every call ("S2").
    - "smw" (default): exploits `E E* = (I_I kron A A*) + B B*` (Eq. (48))
      via the Sherman-Morrison-Woodbury identity (Eq. (51)). Only `A` is
      ever factored (once, through a single sparse LU reused for both the
      `A`- and `A*`-solves via conjugate-transpose solves) plus a dense
      `P x P` "capacitance" matrix `S = I_P + sum_i B_i*(A A*)^-1 B_i`
      (Eq. (50)), also factored once ("S3", Sec. 5.4). Never forms or
      factors the `I*L x I*L` matrix `E E*`.
    - "smw_cg": identical block structure, but the capacitance system
      `S c = s` is solved inexactly by conjugate gradient with a decaying
      tolerance schedule `eta_k`, using a fully matrix-free capacitance
      matvec `c -> c + sum_i B_i*((A A*)^-1 (B_i c))` costing `2*I`
      triangular solves (remark, Sec. 5.4), never even forming `S` or the
      dense `N_i = A^-1 B_i` factors used by "smw" ("S4", Sec. 5.5-5.6).

    All backends return the *projected* `(u_list, m)` pair, i.e.
    `x - E*(E E*)^-1 E x`.
    """

    def __init__(
        self,
        A,
        B,
        method="smw",
        cg_eta0=None,
        cg_gamma=0.8,
        cg_min_tol=1e-8,
        cg_maxiter=200,
        logger=None,
    ):
        if method not in ("spsolve", "cached_splu", "smw", "smw_cg"):
            raise ValueError(f"Unknown projector method: {method!r}")
        self.A = A.tocsr()
        self.A_star = self.A.conj().T
        self.B = [Bi.tocsr() for Bi in B]
        self.B_star = [Bi.conj().T for Bi in self.B]
        self.I = len(B)
        self.L = A.shape[0]
        self.P = B[0].shape[1]
        self.method = method
        self.logger = logger
        self.n_calls = 0
        self.inner_iterations = []
        # Work counters, so a caller can compare algorithms at equal linear
        # algebra budget rather than equal iteration count (Sec. 4.8). An
        # "A-solve" is one triangular solve against the cached LU of `A` or
        # of `A^H`; `n_A_matvecs` counts applications of the sparse `A`/`A^*`
        # themselves, which are an order of magnitude cheaper.
        self.n_A_solves = 0
        self.n_A_matvecs = 0
        # Lazily built eigendecomposition of `H = sum_i N_i^H N_i`, used only
        # by the weighted projection below (see `project`). Building it costs
        # one dense Hermitian eigendecomposition of a P x P matrix, after
        # which `(c I + H)^-1` is available for *any* `c` at O(P^2), which is
        # what makes a per-iteration-varying block metric affordable.
        self._cap_eig = None
        self.cg_eta0 = cg_eta0
        self.cg_gamma = cg_gamma
        # The residual-to-projection bound (Eq. 53) only needs a summable
        # tolerance schedule, not machine precision; as k grows, eta_k =
        # eta0*gamma^k decays geometrically and would otherwise eventually
        # demand an unreasonably (or, once it underflows, impossibly) tight
        # CG solve every single outer iteration. `cg_min_tol` floors the
        # *relative* tolerance (the report's own Eq. 54 remark: "floored at
        # eta_k >= eps_mach ||r^k||", though we use a looser, more practical
        # floor by default), and `cg_maxiter` hard-caps inner iterations regardless,
        # since S = I_P + sum_i N_i^H N_i is provably well-conditioned
        # (Sec. 5.4: S >= I_P) so CG should never need many iterations to
        # reach a sane tolerance; hitting the cap signals the schedule is
        # asking for more accuracy than is useful and is logged as such.
        self.cg_min_tol = cg_min_tol
        self.cg_maxiter = cg_maxiter

        t0 = time.time()
        if method in ("smw", "smw_cg"):
            # Single factorization of A, reused for A- and A*-solves alike
            # (trans="H" solves the conjugate-transpose system), Sec. 5.4.
            self.A_lu = sp.linalg.splu(self.A.tocsc())
            if method == "smw":
                # Capacitance matrix S = I_P + sum_i N_i^H N_i, N_i = A^-1 B_i
                # (Eq. (50)), assembled and factored once.
                S = np.eye(self.P, dtype=complex)
                for Bi in self.B:
                    Ni = self.A_lu.solve(Bi.toarray())
                    S += Ni.conj().T @ Ni
                self.capacitance = S
                self.cap_lu_piv = sla.lu_factor(S)
        else:  # "spsolve" / "cached_splu": assemble E E* explicitly (Eq. 48)
            self.A_block = sp.block_diag([self.A] * self.I, format="csr")
            self.B_stacked = sp.vstack(self.B, format="csr")
            if method == "cached_splu":
                EE_star = (
                    self.A_block @ self.A_block.conj().T
                    + self.B_stacked @ self.B_stacked.conj().T
                )
                self.EE_star_lu = sp.linalg.splu(EE_star.tocsc())
        self.setup_time = time.time() - t0
        if self.logger:
            self.logger.info(
                f"AffineConstraintProjector[{method}] setup in {self.setup_time:.4f}s"
            )

    def feasibility_residual_norm(self, u_list, m):
        """||E x||_2 = sqrt(sum_i ||A u_i - B_i m||_2^2), the PDE-constraint
        violation; used purely as a diagnostic (it is ~0 by construction
        right after `project`, up to the inner-solve tolerance)."""
        return float(
            np.sqrt(
                sum(
                    np.linalg.norm(self.A @ u_list[i] - self.B[i] @ m) ** 2
                    for i in range(self.I)
                )
            )
        )

    def _capacitance_matvec(self, c, shift=1.0):
        # c -> shift*c + sum_i N_i^H N_i c, computed via 2 triangular solves
        # per source without ever forming N_i or S (matrix-free, Sec. 5.4
        # remark). `shift` is 1 for the unweighted projection and
        # `tau_u / tau_m` for the block-weighted one.
        total = shift * c
        for i in range(self.I):
            t = self.A_lu.solve(self.B[i] @ c)
            h = self.A_lu.solve(t, trans="H")
            self.n_A_solves += 2
            total = total + self.B_star[i] @ h
        return total

    def _shifted_capacitance_solve(self, s, shift):
        """Solve `(shift * I_P + H) c = s` with `H = sum_i N_i^H N_i`, for a
        `shift` that may change at every outer iteration.

        `H` is Hermitian positive semi-definite, so one eigendecomposition
        `H = V diag(w) V^H` (built once, on first use) turns every subsequent
        shifted solve into `V ((V^H s) / (shift + w))`, i.e. O(P^2) instead of
        a fresh O(P^3) factorization. That is what makes the block metric of
        the accelerated variants free: the shift is `tau_u / tau_m` and moves
        every iteration.
        """
        if self._cap_eig is None:
            H = self.capacitance - np.eye(self.P, dtype=complex)
            w, V = np.linalg.eigh(H)
            # H is PSD in exact arithmetic; clip the O(eps) negative tail so a
            # small `shift` cannot produce a spurious near-singular mode.
            self._cap_eig = (np.maximum(w.real, 0.0), V, V.conj().T.copy())
        w, V, V_h = self._cap_eig
        return V @ ((V_h @ s) / (shift + w))

    def project(self, u_list, m, iteration=None, weight_ratio=1.0):
        """Project `(u_list, m)` onto `{A u_i = B_i m}`.

        Args:
            weight_ratio: `c = tau_u / tau_m`, the ratio of the primal step
                sizes of the field block and the contrast block. `c = 1` (the
                default) is the Euclidean projection every existing caller
                wants, and takes exactly the code path it always took. For
                `c != 1` this returns the projection in the metric
                `T^-1 = diag(tau_u^-1 I, tau_m^-1 I)`, i.e.
                `prox^T_{iota_C}(z) = z - T E^*(E T E^*)^-1 E z`, which is the
                primal step of a Chambolle-Pock iteration carrying a *block*
                primal metric. Two things change relative to `c = 1`: the
                capacitance system becomes `(c I_P + H) c = s` instead of
                `(I_P + H) c = s`, and the contrast update picks up a `1/c`.
                Only the "smw" and "smw_cg" backends support it; the two
                baseline backends assemble `E E^*` explicitly and would have
                to reassemble and refactor it on every call.
        """
        self.n_calls += 1
        if weight_ratio != 1.0 and self.method in ("spsolve", "cached_splu"):
            raise NotImplementedError(
                f"AffineConstraintProjector[{self.method}]: weight_ratio != 1 needs "
                "the SMW block structure (the capacitance shift). Use "
                "method='smw' or 'smw_cg'."
            )
        if self.method == "spsolve":
            EE_star = (
                self.A_block @ self.A_block.conj().T
                + self.B_stacked @ self.B_stacked.conj().T
            )
            r = np.concatenate(
                [self.A @ u_list[i] - self.B[i] @ m for i in range(self.I)]
            )
            w = sp.linalg.spsolve(EE_star.tocsc(), r)
            w_list = [w[i * self.L : (i + 1) * self.L] for i in range(self.I)]
        elif self.method == "cached_splu":
            r = np.concatenate(
                [self.A @ u_list[i] - self.B[i] @ m for i in range(self.I)]
            )
            w = self.EE_star_lu.solve(r)
            w_list = [w[i * self.L : (i + 1) * self.L] for i in range(self.I)]
        else:  # "smw" / "smw_cg"
            r_list = [self.A @ u_list[i] - self.B[i] @ m for i in range(self.I)]
            self.n_A_matvecs += self.I
            g_list = []
            for ri in r_list:
                t = self.A_lu.solve(ri)
                g_list.append(self.A_lu.solve(t, trans="H"))
                self.n_A_solves += 2
            s = sum(self.B_star[i] @ g_list[i] for i in range(self.I))

            if self.method == "smw":
                c = (
                    sla.lu_solve(self.cap_lu_piv, s)
                    if weight_ratio == 1.0
                    else self._shifted_capacitance_solve(s, weight_ratio)
                )
            else:  # "smw_cg": inexact capacitance solve, Sec. 5.5-5.6
                eta0 = (
                    self.cg_eta0
                    if self.cg_eta0 is not None
                    else max(np.linalg.norm(s), 1.0)
                )
                k = iteration if iteration is not None else 0
                eta_k = max(
                    eta0 * (self.cg_gamma**k),
                    self.cg_min_tol * max(np.linalg.norm(s), 1.0),
                )
                n_iter = [0]

                def _count(_):
                    n_iter[0] += 1

                cap_op = sp.linalg.LinearOperator(
                    (self.P, self.P),
                    matvec=lambda vec: self._capacitance_matvec(vec, weight_ratio),
                    dtype=complex,
                )
                c, info = sp.linalg.cg(
                    cap_op,
                    s,
                    rtol=eta_k / max(np.linalg.norm(s), 1e-300),
                    atol=0.0,
                    maxiter=self.cg_maxiter,
                    callback=_count,
                )
                self.inner_iterations.append(n_iter[0])
                if self.logger and info != 0:
                    self.logger.warning(
                        f"AffineConstraintProjector: CG did not converge (info={info}) "
                        f"after {n_iter[0]} iterations at outer iteration {iteration}."
                    )

            w_list = []
            for i in range(self.I):
                t_prime = self.A_lu.solve(self.B[i] @ c)
                h = self.A_lu.solve(t_prime, trans="H")
                self.n_A_solves += 2
                w_list.append(g_list[i] - h)

        u_new = [u_list[i] - self.A_star @ w_list[i] for i in range(self.I)]
        self.n_A_matvecs += self.I
        correction_m = sum(self.B_star[i] @ w_list[i] for i in range(self.I))
        m_new = m + (correction_m if weight_ratio == 1.0 else correction_m / weight_ratio)
        return u_new, m_new


class ProjectedChambollePock(FixedPointAlgorithm):
    """Inexact Projected Chambolle-Pock algorithm (Algorithm 5, Sec. 5.5): the
    complementary instantiation to `ChambollePock` (Algorithm 3). Here the
    PDE constraint is kept in the *primal* proximable block `f = iota_C`,
    evaluated by the closed-form (exact or inexact) affine projection of
    Sec. 5.3-5.4 through an `AffineConstraintProjector`, while the
    data-fidelity term and the regularizer on m (TV or Tikhonov) are
    *dualized* (Eq. (43)-(45)). That is the reverse of Algorithm 3, where the
    PDE constraint is dualized and the data term is proxed. The key practical
    payoff (Sec. 5.1, 5.6) is that the step-size condition no longer involves
    `||A||`, only `||C||` and the regularizer operator's norm, making it far
    less sensitive to mesh refinement than `ChambollePock`.

    State layout is the same as `ChambollePock`: `x = [u_0, ..., u_{I-1}, m]`.
    `sigma_dat` and `sigma_reg` are the (block-preconditioned, Eq. (56))
    dual step sizes for the data and regularizer blocks respectively; a
    single scalar `sigma` can be passed to both for the non-preconditioned
    variant (Eq. (33)).

    Two optional accelerations sit behind flags whose defaults leave the
    iteration exactly as it was (Sec. 4.8):

    `accelerate`
        ``"none"`` (default) is the plain scheme. ``"subspace"`` and
        ``"dual_data"`` are the two halves of the partial acceleration of
        Valkonen and Pock, *Acceleration of the PDHGM on strongly convex
        subspaces* (arXiv:1511.06566), which handles objectives that are
        strongly convex only on part of the space by giving each block its
        own step size and accelerating only the block that earns it. What
        makes that free here is a structural accident worth stating: the two
        dual blocks act on *disjoint* primal variables, the data block on the
        fields `u_i` and the regularizer block on the contrast `m`, so
        `K^* Sigma K` is block diagonal and the step-size condition splits
        into two independent conditions

            tau_u * sigma_dat * ||C||^2 <= 1,   tau_m * sigma_reg * ||G||^2 <= 1

        (the same "max, not sum" observed in `block_step_sizes_algorithm5`).
        Either pair can therefore be rescheduled without touching the other.

        ``"subspace"`` moves the Tikhonov term `(gamma/2)||m||^2` out of the
        dual and into the primal contrast block, where it is exact, and then
        runs the adaptive rule of Algorithm 2 of Chambolle-Pock (2011) on
        that block alone::

            theta_k     = 1 / sqrt(1 + 2 gamma tau_m,k)
            tau_m,k+1   = theta_k tau_m,k
            sigma_reg,k+1 = sigma_reg,k / theta_k

        with the field block left at its fixed `(tau_u, sigma_dat)`. `gamma`
        is the model's own `mu`, known exactly, never estimated. Note that
        for `reg_mode="tikhonov"` moving the quadratic to the primal removes
        the only dual block acting on `m`, which leaves `tau_m` out of the
        step-size condition altogether: there is then nothing to schedule and
        the rule is skipped (see `accel_note`), the gain coming entirely from
        handling the strongly convex block exactly rather than through a dual
        variable. The schedule is live only when a dual block on `m`
        survives, i.e. TV (or first-order Tikhonov) plus a zeroth-order
        Tikhonov term.

        ``"dual_data"`` accelerates the other block. `g*_dat(v) =
        (1/2)||v||^2 + Re<v,d>` is 1-strongly convex, so Algorithm 2 applies
        to the *dual* of the data block: the roles of the two updates swap
        (primal first, extrapolation carried by the dual variable) and

            theta_k       = 1 / sqrt(1 + 2 gamma sigma_dat,k)
            sigma_dat,k+1 = theta_k sigma_dat,k
            tau_u,k+1     = tau_u,k / theta_k

        with `sigma_reg` and `tau_m` held fixed, which is what keeps the
        inexactness analysis of Sec. 5.4 (an argument about the primal
        projection) untouched.

        ``"dual_both"`` is not in the Sec. 4.8 plan; it is what measuring
        ``"dual_data"`` pointed at. Holding `tau_m` fixed while `tau_u` grows
        like `k` drives the metric ratio `tau_u/tau_m` up without bound, and
        since the projection's contrast correction carries a factor
        `tau_m/tau_u`, the contrast block freezes: the scheme is fast to
        moderate accuracy and then stalls. The way out is not a smaller
        schedule but the observation that under a *quadratic* regularizer
        there is nothing partial about the strong convexity in the first
        place. Dualizing `(mu/2)||m||^2` gives `g*_reg(v) = ||v||^2/(2 mu)`,
        of modulus `1/mu`, alongside the data block's modulus 1, so the whole
        of `g*` is strongly convex with modulus `gamma = min(1, 1/mu)` and
        plain Algorithm 2 applies to both blocks at once::

            theta_k = 1 / sqrt(1 + 2 gamma min(sigma_dat,k, sigma_reg,k))
            sigma_.,k+1 = theta_k sigma_.,k,   tau_.,k+1 = tau_.,k / theta_k

        Both primal steps grow together, so the metric ratio stays at 1 and
        the projection keeps its cheap unweighted path. This is available for
        the Tikhonov modes only: under TV, `g*_reg` is the indicator of the
        l2,inf ball, which is not strongly convex, and the mode raises.

    `linesearch`
        Malitsky and Pock, *A first-order primal-dual algorithm with
        linesearch*, SIAM J. Optim. 28(1):411-432, 2018 (Algorithm 4). The
        primal step is taken first, then only the *dual* update is repeated
        in a backtracking loop until

            ||K^*(v_k - v_{k-1})||_T <= delta_ls ||v_k - v_{k-1}||_{Sigma^-1}

        holds, the block-metric form of their scalar test. `||K||` is never
        needed. A rejected trial costs one dual update, i.e. `I` applications
        of `C`, against a primal step costing `4I` triangular solves against
        `A`, so backtracking is cheap in the currency that actually matters
        here.
    """

    ACCELERATION_MODES = ("none", "subspace", "dual_data", "dual_both")

    def __init__(
        self,
        exp_name,
        algo_plot_name,
        f,
        C,
        d,
        I,
        L,
        P,
        tau,
        sigma_dat,
        sigma_reg,
        projector,
        reg_mode="tikhonov",
        mu=None,
        G=None,
        lambda_tv=None,
        accelerate="none",
        gamma=None,
        tau_m=None,
        tau_max=None,
        linesearch=False,
        mu_ls=0.7,
        delta_ls=0.99,
        ls_max_trials=60,
        logger=None,
        verbose=False,
    ):
        super().__init__(exp_name, algo_plot_name, f, logger=logger, verbose=verbose)
        self.C = C
        self.C_star = C.conj().T
        self.d = d
        self.I = I
        self.L = L
        self.P = P
        self.tau = tau
        self.sigma_dat = sigma_dat
        self.sigma_reg = sigma_reg
        self.projector = projector
        self.reg_mode = reg_mode
        self.G = G
        self.G_star = G.conj().T if G is not None else None

        if reg_mode == "tv":
            if G is None or lambda_tv is None:
                raise ValueError("reg_mode='tv' requires both G and lambda_tv")
            self.prox_reg_dual = make_tv_dual_prox(lambda_tv)
            Q = G.shape[0]
        elif reg_mode == "tikhonov":
            if mu is None:
                raise ValueError("reg_mode='tikhonov' requires mu")
            self.prox_reg_dual = make_tikhonov_dual_prox(mu)
            Q = P
        elif reg_mode == "tikhonov1":
            # First-order Tikhonov, g(m) = (mu/2)||G m||^2, dualized through
            # G exactly as TV is. With `G = pb.G_h1` this is the discrete H^1
            # seminorm; the dual prox is the same resolvent as the
            # zeroth-order case, only composed with G.
            if G is None or mu is None:
                raise ValueError("reg_mode='tikhonov1' requires both G and mu")
            self.prox_reg_dual = make_tikhonov_dual_prox(mu)
            Q = G.shape[0]
        elif reg_mode == "none":
            self.prox_reg_dual = None
            Q = 0
        else:
            raise ValueError(f"Unknown reg_mode: {reg_mode!r}")

        if accelerate not in self.ACCELERATION_MODES:
            raise ValueError(
                f"Unknown accelerate: {accelerate!r} "
                f"(expected one of {self.ACCELERATION_MODES})"
            )
        self.accelerate = accelerate
        self.linesearch = bool(linesearch)
        if self.linesearch and accelerate in ("dual_data", "dual_both"):
            raise ValueError(
                "linesearch=True is incompatible with accelerate='dual_data': the "
                "line search searches the primal step tau, while dual_data drives "
                "tau by its own schedule. Combining them would leave neither rule's "
                "hypotheses satisfied."
            )
        if not (0.0 < mu_ls < 1.0):
            raise ValueError(f"mu_ls must lie in (0,1), got {mu_ls}")
        if not (0.0 < delta_ls < 1.0):
            raise ValueError(f"delta_ls must lie in (0,1), got {delta_ls}")
        self.mu_ls = mu_ls
        self.delta_ls = delta_ls
        self.ls_max_trials = int(ls_max_trials)
        # Ceiling on the primal step, i.e. a floor on the dual step. The dual
        # schedules are unbounded by construction (sigma ~ 1/(gamma k),
        # tau ~ k), and once tau/sigma has opened up by more than about
        # 1/sqrt(eps_mach) the primal step x - tau K^* v is formed by
        # cancelling two large numbers, which puts a floor under the
        # attainable accuracy that has nothing to do with the convergence
        # rate. Freezing the schedule once tau reaches `tau_max` keeps the
        # early acceleration and removes the floor. None means unbounded,
        # which is the schedule exactly as written in the papers.
        self.tau_max = None if tau_max is None else float(tau_max)
        self.schedule_frozen_at = None

        # --- block primal steps. `tau_m=None` means "same as tau", which is
        # the historical single-step behaviour and keeps the projection
        # weight ratio at exactly 1.0, i.e. the untouched code path.
        self.tau_u = tau
        self.tau_m = tau if tau_m is None else tau_m
        self.gamma_primal = 0.0  # modulus of the Tikhonov term kept primal
        self.gamma_dual = 0.0  # modulus of strong convexity of g*_dat
        self.accel_note = ""
        self._dual_reg_active = reg_mode != "none"
        self._schedule_live = False

        if accelerate == "subspace":
            if reg_mode == "tikhonov":
                if gamma is not None and gamma != mu:
                    raise ValueError(
                        "accelerate='subspace' with reg_mode='tikhonov' moves the "
                        f"whole Tikhonov term into the primal, so gamma must equal mu "
                        f"({mu!r}); got {gamma!r}. Passing a different modulus would "
                        "change the problem, not the algorithm."
                    )
                self.gamma_primal = float(mu)
                # The quadratic has left the dual: no dual block acts on m.
                self._dual_reg_active = False
                self.accel_note = (
                    "tikhonov: quadratic moved to the primal block, which removes the "
                    "only dual block on m, so tau_m is unconstrained and the "
                    "theta-schedule is vacuous (held fixed)."
                )
            elif reg_mode in ("tv", "tikhonov1"):
                if gamma is None:
                    raise ValueError(
                        f"accelerate='subspace' with reg_mode={reg_mode!r} needs an "
                        "explicit gamma: the m-block of "
                        + (
                            "a TV objective is not strongly convex"
                            if reg_mode == "tv"
                            else "a first-order Tikhonov objective is only strongly "
                            "convex modulo constants"
                        )
                        + ", so acceleration requires an added zeroth-order Tikhonov "
                        "term (gamma/2)||m||^2. Pass gamma to add it explicitly."
                    )
                self.gamma_primal = float(gamma)
                self._schedule_live = True
                self.accel_note = (
                    f"{reg_mode}: added primal Tikhonov term gamma={gamma:g}; "
                    "theta-schedule live on (tau_m, sigma_reg)."
                )
            else:
                raise ValueError(
                    "accelerate='subspace' needs a regularizer on m; got "
                    f"reg_mode={reg_mode!r}."
                )
        elif accelerate == "dual_data":
            # g*_dat(v) = (1/2)||v||^2 + Re<v,d> has modulus exactly 1.
            self.gamma_dual = 1.0 if gamma is None else float(gamma)
            self._schedule_live = True
            self.accel_note = (
                f"dual_data: theta-schedule live on (sigma_dat, tau_u), "
                f"gamma={self.gamma_dual:g}."
            )
        elif accelerate == "dual_both":
            if reg_mode == "tv":
                raise ValueError(
                    "accelerate='dual_both' needs every dual block to be strongly "
                    "convex, and the TV block's conjugate is the indicator of the "
                    "l2,inf ball, which is not. Use accelerate='dual_data' (which "
                    "accelerates only the data block) or a Tikhonov reg_mode."
                )
            if gamma is not None:
                self.gamma_dual = float(gamma)
            elif reg_mode == "none":
                self.gamma_dual = 1.0
            else:
                # min over blocks: 1 for the data block, 1/mu for the dualized
                # quadratic. mu < 1 throughout here, so the data block binds.
                self.gamma_dual = min(1.0, 1.0 / float(mu))
            self._schedule_live = True
            self.accel_note = (
                f"dual_both: theta-schedule live on both dual blocks, "
                f"gamma={self.gamma_dual:g}; the metric ratio stays at 1."
            )

        # Fixed dual/primal ratios, held invariant by the line search so that
        # backtracking rescales the whole block metric rather than distorting
        # the preconditioning it was given.
        self._beta_dat = sigma_dat / self.tau_u
        self._beta_reg = sigma_reg / self.tau_m
        self._theta_prev = 1.0

        self.data_dual_prox = [make_data_dual_prox(d[i]) for i in range(I)]
        self.v_dat = [np.zeros(C.shape[0], dtype=complex) for _ in range(I)]
        self.v_reg = np.zeros(Q, dtype=complex)
        self.v_bar_dat = None
        self.v_bar_reg = None
        self.x_bar = None
        self.current_residual = None
        self.current_step_norm = None

        # Reporting: step-size trajectories and work counters.
        self.tau_u_history = []
        self.tau_m_history = []
        self.sigma_dat_history = []
        self.sigma_reg_history = []
        self.ls_trials = []
        self.n_C_applies = 0

    def _split(self, x):
        u = [x[i * self.L : (i + 1) * self.L] for i in range(self.I)]
        m = x[self.I * self.L : self.I * self.L + self.P]
        return u, m

    def _assemble(self, x_like, u, m):
        out = np.empty_like(x_like)
        for i in range(self.I):
            out[i * self.L : (i + 1) * self.L] = u[i]
        out[self.I * self.L : self.I * self.L + self.P] = m
        return out

    # -- the two half-steps, shared by all three iteration orderings --------

    def _dual_update(self, u_bar, m_bar, sigma_dat, sigma_reg):
        """One dual update from the stored `v_dat`/`v_reg`. Never mutates
        state, so the line search can call it repeatedly on trial steps."""
        v_dat_new = [
            self.data_dual_prox[i](
                self.v_dat[i] + sigma_dat * (self.C @ u_bar[i]), sigma_dat
            )
            for i in range(self.I)
        ]
        self.n_C_applies += self.I
        if not self._dual_reg_active:
            return v_dat_new, self.v_reg
        if self.reg_mode in ("tv", "tikhonov1"):
            v_reg_new = self.prox_reg_dual(
                self.v_reg + sigma_reg * (self.G @ m_bar), sigma_reg
            )
        else:  # "tikhonov", dualized through the identity
            v_reg_new = self.prox_reg_dual(self.v_reg + sigma_reg * m_bar, sigma_reg)
        return v_dat_new, v_reg_new

    def _primal_update(self, u, m, v_dat, v_reg):
        """`prox^T_{tau f}(x - T K^* v)` in the block metric
        `T = diag(tau_u I, tau_m I)`, with the Tikhonov term folded in when
        `accelerate="subspace"` keeps it primal."""
        z_u = [u[i] - self.tau_u * (self.C_star @ v_dat[i]) for i in range(self.I)]
        self.n_C_applies += self.I
        if not self._dual_reg_active:
            z_m = m.copy()
        elif self.reg_mode in ("tv", "tikhonov1"):
            z_m = m - self.tau_m * (self.G_star @ v_reg)
        else:
            z_m = m - self.tau_m * v_reg
        weight_ratio = self.tau_u / self.tau_m
        if self.gamma_primal:
            # (1/2tau_m)||m-z_m||^2 + (gamma/2)||m||^2 is, up to a constant,
            # ((1+tau_m gamma)/2tau_m)||m - z_m/(1+tau_m gamma)||^2: the same
            # weighted projection with a shrunk target and a rescaled metric.
            shrink = 1.0 + self.tau_m * self.gamma_primal
            z_m = z_m / shrink
            weight_ratio = weight_ratio * shrink
        return self.projector.project(
            z_u, z_m, iteration=self.iteration, weight_ratio=weight_ratio
        )

    def _record(self):
        self.tau_u_history.append(self.tau_u)
        self.tau_m_history.append(self.tau_m)
        self.sigma_dat_history.append(self.sigma_dat)
        self.sigma_reg_history.append(self.sigma_reg)

    def _finish(self, x, u_new, m_new):
        x_new = self._assemble(x, u_new, m_new)
        self.current_residual = self.projector.feasibility_residual_norm(u_new, m_new)
        self.current_step_norm = np.linalg.norm(x_new - x)
        self._record()
        return x_new

    # -- the three iteration orderings -------------------------------------

    def step(self, x):
        if self.linesearch:
            return self._step_linesearch(x)
        if self.accelerate in ("dual_data", "dual_both"):
            return self._step_dual_extrapolated(x)
        return self._step_primal_extrapolated(x)

    def _step_primal_extrapolated(self, x):
        """Dual update, primal update, primal extrapolation: the original
        Algorithm 5 ordering. With `accelerate="none"` this is bit-for-bit
        the iteration this class has always run."""
        u, m = self._split(x)
        if self.x_bar is None:
            u_bar, m_bar = u, m
        else:
            u_bar, m_bar = self._split(self.x_bar)

        v_dat_new, v_reg_new = self._dual_update(
            u_bar, m_bar, self.sigma_dat, self.sigma_reg
        )
        u_new, m_new = self._primal_update(u, m, v_dat_new, v_reg_new)

        theta_m = 1.0
        if self.accelerate == "subspace" and self._schedule_live:
            theta_m = 1.0 / np.sqrt(1.0 + 2.0 * self.gamma_primal * self.tau_m)

        if self.x_bar is None:
            self.x_bar = np.empty_like(x)
        for i in range(self.I):
            # theta = 1 is written as 2a - b so the default path is
            # bit-for-bit what it was before the flag existed.
            self.x_bar[i * self.L : (i + 1) * self.L] = 2 * u_new[i] - u[i]
        m_slice = slice(self.I * self.L, self.I * self.L + self.P)
        self.x_bar[m_slice] = (
            2 * m_new - m if theta_m == 1.0 else m_new + theta_m * (m_new - m)
        )

        if theta_m != 1.0:
            self.tau_m = self.tau_m * theta_m
            self.sigma_reg = self.sigma_reg / theta_m

        self.v_dat = v_dat_new
        self.v_reg = v_reg_new
        return self._finish(x, u_new, m_new)

    def _step_dual_extrapolated(self, x):
        """Primal update, dual update, *dual* extrapolation: Algorithm 2 of
        Chambolle-Pock (2011) applied to the dual, which is the form that
        exploits strong convexity of `g*` rather than of `f`."""
        u, m = self._split(x)
        if self.v_bar_dat is None:
            self.v_bar_dat = [v.copy() for v in self.v_dat]
            self.v_bar_reg = self.v_reg.copy()

        u_new, m_new = self._primal_update(u, m, self.v_bar_dat, self.v_bar_reg)
        v_dat_new, v_reg_new = self._dual_update(
            u_new, m_new, self.sigma_dat, self.sigma_reg
        )

        both = self.accelerate == "dual_both"
        if self.tau_max is not None and self.tau_u >= self.tau_max:
            if self.schedule_frozen_at is None:
                self.schedule_frozen_at = self.iteration
                if self.logger:
                    self.logger.info(
                        f"{self.algo_plot_name}: step-size schedule frozen at "
                        f"iteration {self.iteration}, tau_u={self.tau_u:.4g}"
                    )
            self.v_bar_dat = [2 * v_dat_new[i] - self.v_dat[i] for i in range(self.I)]
            self.v_bar_reg = 2 * v_reg_new - self.v_reg
            self.v_dat = v_dat_new
            self.v_reg = v_reg_new
            return self._finish(x, u_new, m_new)

        sigma_ref = (
            min(self.sigma_dat, self.sigma_reg)
            if both and self._dual_reg_active
            else self.sigma_dat
        )
        theta = 1.0 / np.sqrt(1.0 + 2.0 * self.gamma_dual * sigma_ref)
        self.v_bar_dat = [
            v_dat_new[i] + theta * (v_dat_new[i] - self.v_dat[i]) for i in range(self.I)
        ]
        if both:
            self.v_bar_reg = v_reg_new + theta * (v_reg_new - self.v_reg)
            self.sigma_reg = self.sigma_reg * theta
            self.tau_m = self.tau_m / theta
        else:
            # Under "dual_data" the regularizer block is not assumed strongly
            # convex, so it keeps theta = 1 and its steps: that asymmetry is
            # the whole content of the partial acceleration.
            self.v_bar_reg = 2 * v_reg_new - self.v_reg

        self.sigma_dat = self.sigma_dat * theta
        self.tau_u = self.tau_u / theta

        self.v_dat = v_dat_new
        self.v_reg = v_reg_new
        return self._finish(x, u_new, m_new)

    def _step_linesearch(self, x):
        """Malitsky-Pock Algorithm 4. The primal step uses the step accepted
        at the previous iteration; only the dual update is repeated."""
        u, m = self._split(x)
        u_new, m_new = self._primal_update(u, m, self.v_dat, self.v_reg)

        tau_u_prev, tau_m_prev = self.tau_u, self.tau_m
        scale = np.sqrt(1.0 + self._theta_prev)
        v_dat_new = v_reg_new = None
        theta = 1.0
        accepted = False
        for trial in range(1, self.ls_max_trials + 1):
            # `theta` is the scale the trial duals were actually computed at.
            # Keeping it separate from `scale` matters on the path where the
            # loop runs out of trials: `scale` has been shrunk once more by
            # then, and pairing the accepted duals with a step size they were
            # not computed at would silently break the metric.
            theta = scale
            tau_u, tau_m = tau_u_prev * scale, tau_m_prev * scale
            sigma_dat = self._beta_dat * tau_u
            sigma_reg = self._beta_reg * tau_m
            u_bar = [u_new[i] + theta * (u_new[i] - u[i]) for i in range(self.I)]
            m_bar = m_new + theta * (m_new - m)
            v_dat_new, v_reg_new = self._dual_update(u_bar, m_bar, sigma_dat, sigma_reg)

            dv_dat = [v_dat_new[i] - self.v_dat[i] for i in range(self.I)]
            lhs = tau_u * sum(
                np.linalg.norm(self.C_star @ dv) ** 2 for dv in dv_dat
            )
            self.n_C_applies += self.I
            rhs = sum(np.linalg.norm(dv) ** 2 for dv in dv_dat) / sigma_dat
            if self._dual_reg_active:
                dv_reg = v_reg_new - self.v_reg
                k_star_dv_m = (
                    self.G_star @ dv_reg
                    if self.reg_mode in ("tv", "tikhonov1")
                    else dv_reg
                )
                lhs = lhs + tau_m * np.linalg.norm(k_star_dv_m) ** 2
                rhs = rhs + np.linalg.norm(dv_reg) ** 2 / sigma_reg
            if lhs <= (self.delta_ls**2) * rhs or rhs == 0.0:
                accepted = True
                break
            scale = scale * self.mu_ls
        self.ls_trials.append(trial)
        if not accepted and self.logger:
            self.logger.warning(
                f"{self.algo_plot_name}: line search hit its trial cap "
                f"({self.ls_max_trials}) at iteration {self.iteration}; accepting "
                f"tau={tau_u_prev * theta:.4g} without the descent test."
            )

        self.tau_u, self.tau_m = tau_u_prev * theta, tau_m_prev * theta
        self.sigma_dat = self._beta_dat * self.tau_u
        self.sigma_reg = self._beta_reg * self.tau_m
        self._theta_prev = theta
        self.v_dat = v_dat_new
        self.v_reg = v_reg_new
        return self._finish(x, u_new, m_new)

    def is_converged(self, x, threshold=1e-6):
        # ChambollePock/DistributedChambollePock track the (asymptotic) PDE
        # feasibility gap as their convergence proxy; here that gap is ~0 at
        # every iteration by construction (Sec. 5.1), so we instead use the
        # fixed-point step size ||x_{k+1} - x_k||, a standard practical
        # stopping rule for primal-dual algorithms without a cheap duality
        # gap (see the notebook's discussion of this choice).
        #
        # The two primal-first orderings (dual_data and the line search) take
        # the primal step before the dual one, so from the standard start
        # x0 = 0, v0 = 0 their first iterate is prox_f(x0) = x0 and the step
        # norm is exactly zero: the rule would declare convergence before the
        # algorithm has done anything. Skip the first step for those. The
        # default ordering updates the dual first and never has a null first
        # step, so its behaviour is untouched.
        if self.current_step_norm is None:
            return False
        if (self.linesearch or self.accelerate in ("dual_data", "dual_both")) and (
            self.iteration or 0
        ) < 2:
            return False
        return self.current_step_norm < threshold


### Distributed Algorithms ###


class DistributedAlgorithm(FixedPointAlgorithm, abc.ABC):
    """Minimal base class for distributed algorithms.

    Keeps the same constructor signature as `FixedPointAlgorithm` and only
    adds optional MPI-related arguments. Algorithm-specific distributed
    parameters (sizes, matrices, callbacks) belong on concrete subclasses.
    """

    def __init__(
        self,
        exp_name,
        algo_plot_name,
        f,
        logger=None,
        verbose=False,
        agent_indices=None,
        use_mpi=False,
        mpi_comm=None,
        local_agents=None,
    ):
        super().__init__(exp_name, algo_plot_name, f, logger=logger, verbose=verbose)
        self.use_mpi = use_mpi
        self.mpi_comm = None
        self.rank = 0
        self.world_size = 1
        if self.use_mpi:
            try:
                from mpi4py import MPI

                self.mpi_comm = mpi_comm or MPI.COMM_WORLD
                self.rank = self.mpi_comm.Get_rank()
                self.world_size = self.mpi_comm.Get_size()
            except Exception:
                raise ImportError("mpi4py is required when use_mpi=True")

        self.local_agents = list(local_agents) if local_agents is not None else None

        # Optional agent -> local block assignment mapping provided by caller.
        # If given, build an index -> agent mapping and derive sensible
        # `local_agents` defaults when possible (round-robin per MPI rank).
        self.agent_indices = (
            [list(indices) for indices in agent_indices]
            if agent_indices is not None
            else None
        )
        self.index_to_agent = {}
        if self.agent_indices is not None:
            for s, indices in enumerate(self.agent_indices):
                for i in indices:
                    self.index_to_agent[i] = s
            if self.local_agents is None:
                if self.use_mpi:
                    self.local_agents = [
                        s
                        for s in range(len(self.agent_indices))
                        if s % self.world_size == self.rank
                    ]
                else:
                    self.local_agents = list(range(len(self.agent_indices)))


class DistributedBlockGradientDescent(DistributedAlgorithm):
    """Distributed block gradient descent assuming a single global `m`.

    State layout: x = [u_0, ..., u_{I-1}, m] with length I*L + P.
    `grad_u(i, u_i, m)` should return gradient w.r.t. u_i (shape L).
    `grad_m(s, m, u_list)` should return the local contribution to the
    gradient of `m` from agent `s` (shape P). The full gradient is the sum
    over agents, and we apply a single gradient step on `m` using that sum.
    """

    def __init__(
        self,
        exp_name,
        algo_plot_name,
        f,
        grad_u,
        grad_m,
        S,
        W,
        alpha,
        P,
        L,
        I=None,
        agent_indices=None,
        use_mpi=False,
        mpi_comm=None,
        local_agents=None,
        logger=None,
        verbose=False,
    ):
        super().__init__(
            exp_name,
            algo_plot_name,
            f,
            logger=logger,
            verbose=verbose,
            agent_indices=agent_indices,
            use_mpi=use_mpi,
            mpi_comm=mpi_comm,
            local_agents=local_agents,
        )

        self.S = S
        self.W = np.asarray(W) if W is not None else None
        self.alpha = alpha
        self.P = P
        self.L = L
        self.I = I if I is not None else S
        self.grad_u = grad_u
        self.grad_m = grad_m

        if self.local_agents is None:
            if self.use_mpi:
                self.local_agents = [
                    s for s in range(self.S) if s % self.world_size == self.rank
                ]
            else:
                self.local_agents = list(range(self.S))

        self.current_gradient = None

    def split_state(self, x):
        if self.P is None or self.L is None:
            raise ValueError(
                "P and L (parameter and local state sizes) must be provided"
            )
        # Global state layout: (u_0, ..., u_{I-1}, m)
        u = [x[i * self.L : (i + 1) * self.L] for i in range(self.I)]
        m_start = self.I * self.L
        m = x[m_start : m_start + self.P]
        return m, u

    def step(self, x):
        m, u = self.split_state(x)

        # Prepare tilde_u and gradient storage
        tilde_u = [ui.copy() for ui in u]
        grad_u_list = [np.zeros(self.L, dtype=x.dtype) for _ in range(self.I)]

        # Local updates for u (only for blocks whose owning agent is local)
        for i in range(self.I):
            s_idx = self.index_to_agent.get(i, i % self.S)
            if s_idx in self.local_agents:
                gu = self.grad_u(i, u[i], m)
                tilde_u[i] = u[i] - self.alpha * gu
                grad_u_list[i] = gu.ravel()

        # Local contributions to gradient of m
        grad_m_map = {}
        for s in self.local_agents:
            gm = self.grad_m(s, m, u)
            grad_m_map[s] = np.asarray(gm).ravel()

        # Aggregate local contributions to gradient of m into a single vector.
        # Use MPI Allreduce (sum) when available to avoid heavy allgather of dicts.
        local_gm_sum = np.zeros(self.P, dtype=x.dtype)
        for s in self.local_agents:
            local_gm_sum += grad_m_map.get(s, np.zeros(self.P, dtype=x.dtype))
        if self.use_mpi:
            from mpi4py import MPI

            gm_total = self.mpi_comm.allreduce(local_gm_sum, op=MPI.SUM)
        else:
            gm_total = local_gm_sum

        # Single gradient step on m using aggregated gradient
        m_new = m - self.alpha * gm_total

        # Build new state (write into preallocated buffer to avoid extra copies)
        x_new = np.empty_like(x)
        # u blocks first
        for i in range(self.I):
            start = i * self.L
            end = start + self.L
            x_new[start:end] = tilde_u[i]
        # then single m
        m_start = self.I * self.L
        x_new[m_start : m_start + self.P] = m_new
        grad_u_concat = [g.ravel() for g in grad_u_list]
        self.current_gradient = np.concatenate(grad_u_concat + [gm_total])
        return x_new

    def is_converged(self, x, threshold=1e-6):
        return (
            self.current_gradient is not None
            and np.linalg.norm(self.current_gradient) < threshold
        )


class DistributedChambollePock(DistributedAlgorithm):
    """Distributed Chambolle-Pock algorithm with exact consensus (Algorithm 4,
    Sec. 4.7).

    Same state layout and operator convention as `ChambollePock` (Algorithm 3),
    but sources are partitioned across S agents (see `agent_indices` in
    `DistributedAlgorithm`), each holding its own dual variables (`v_pde` per
    source, `v_reg` per agent). The consensus variable m is recovered by
    averaging the agents' local updates (Eq. (37)) and broadcasting the
    result, so a single copy of m is kept in the state vector
    x = [u_0, ..., u_{I-1}, m]. See `ChambollePock` for the meaning of `G`
    and `prox_dual_reg` (the regularizer's dualization operator/proximity
    operator, TV or Tikhonov).

    Regularization weight vs. number of agents S. Eq. (35) sums the
    regularizer once per agent, `sum_s R(m_s)`, with *no* `1/S` factor
    (unlike the Decentralized Gradient Descent formulation of Sec. 4.5,
    Eq. (21), which explicitly divides by S). Since exact consensus (Eq. 37)
    ties every m_s to the same value, this means the *effective* penalty
    applied to the shared m is `S` times what a single-agent (or the
    centralized Algorithm 3) run would apply for the same `mu`/`lambda_tv`.
    To reproduce the same regularized problem as `ChambollePock` with S
    agents, divide the regularization weight passed to `prox_dual_reg` by S
    (verified empirically in the comparison notebook: with S=2 agents,
    passing `mu` unscaled shifts the converged m by ~1e-1 relative to the
    centralized solution, while passing `mu / S` recovers it to ~1e-5).
    """

    def __init__(
        self,
        exp_name,
        algo_plot_name,
        f,
        A,
        B,
        C,
        G,
        d,
        S,
        I,
        L,
        P,
        tau,
        sigma,
        prox_dual_reg=None,
        agent_indices=None,
        use_mpi=False,
        mpi_comm=None,
        local_agents=None,
        logger=None,
        verbose=False,
    ):
        super().__init__(
            exp_name,
            algo_plot_name,
            f,
            logger=logger,
            verbose=verbose,
            agent_indices=agent_indices,
            use_mpi=use_mpi,
            mpi_comm=mpi_comm,
            local_agents=local_agents,
        )
        self.A = A
        self.A_star = A.conj().T
        self.B = B
        self.B_star = [Bi.conj().T for Bi in B]
        self.C = C
        self.C_star = C.conj().T
        self.G = G
        self.G_star = G.conj().T if G is not None else None
        self.d = d
        self.S = S
        self.I = I
        self.L = L
        self.P = P
        self.tau = tau
        self.sigma = sigma
        self.prox_dual_reg = prox_dual_reg

        if self.local_agents is None:
            if self.use_mpi:
                self.local_agents = [
                    s for s in range(self.S) if s % self.world_size == self.rank
                ]
            else:
                self.local_agents = list(range(self.S))

        # One-time setup: factor the Woodbury correction matrix, shared across
        # all sources and agents since C does not depend on i, cf. Eq. (32).
        J = C.shape[0]
        woodbury_matrix = sp.eye(J, format="csc", dtype=complex) + tau * (
            C @ self.C_star
        )
        self.woodbury_lu = sp.linalg.splu(woodbury_matrix.tocsc())

        Q = G.shape[0] if G is not None else 0
        self.v_pde = [np.zeros(L, dtype=complex) for _ in range(I)]
        self.v_reg = {s: np.zeros(Q, dtype=complex) for s in range(S)}
        self.u_bar = None
        self.m_bar = None
        self.current_residual = None

    def _sources_for_agent(self, s):
        if self.agent_indices is not None:
            return self.agent_indices[s]
        return [i for i in range(self.I) if i % self.S == s]

    def _prox_data(self, i, v):
        rhs = self.C_star @ self.d[i] + v / self.tau
        correction = self.C_star @ self.woodbury_lu.solve(self.C @ rhs)
        return self.tau * (rhs - self.tau * correction)

    def step(self, x):
        u = [x[i * self.L : (i + 1) * self.L] for i in range(self.I)]
        m_start = self.I * self.L
        m = x[m_start : m_start + self.P]

        if self.u_bar is None:
            self.u_bar = [ui.copy() for ui in u]
            self.m_bar = m.copy()

        u_new = [ui.copy() for ui in u]
        m_tilde_local_sum = np.zeros(self.P, dtype=x.dtype)

        for s in self.local_agents:
            local_sources = self._sources_for_agent(s)

            for i in local_sources:
                self.v_pde[i] = self.v_pde[i] + self.sigma * (
                    self.A @ self.u_bar[i] - self.B[i] @ self.m_bar
                )
                u_new[i] = self._prox_data(
                    i, u[i] - self.tau * (self.A_star @ self.v_pde[i])
                )

            if self.G is not None:
                self.v_reg[s] = self.prox_dual_reg(
                    self.v_reg[s] + self.sigma * (self.G @ self.m_bar), self.sigma
                )

            local_grad = sum(self.B_star[i] @ self.v_pde[i] for i in local_sources)
            if self.G is not None:
                local_grad = local_grad - self.G_star @ self.v_reg[s]
            m_tilde_local_sum += m + self.tau * local_grad

        # Exact consensus : average the agents' local updates, cf. Eq. (37).
        if self.use_mpi:
            from mpi4py import MPI

            m_tilde_total = self.mpi_comm.allreduce(m_tilde_local_sum, op=MPI.SUM)
        else:
            m_tilde_total = m_tilde_local_sum
        m_new = m_tilde_total / self.S

        # Broadcast + over-relaxation (theta = 1)
        self.m_bar = 2 * m_new - m
        local_residual_sq = 0.0
        for s in self.local_agents:
            for i in self._sources_for_agent(s):
                self.u_bar[i] = 2 * u_new[i] - u[i]
                local_residual_sq += (
                    np.linalg.norm(self.A @ u_new[i] - self.B[i] @ m_new) ** 2
                )

        if self.use_mpi:
            from mpi4py import MPI

            total_residual_sq = self.mpi_comm.allreduce(local_residual_sq, op=MPI.SUM)
        else:
            total_residual_sq = local_residual_sq
        self.current_residual = np.sqrt(total_residual_sq)

        x_new = np.empty_like(x)
        for i in range(self.I):
            x_new[i * self.L : (i + 1) * self.L] = u_new[i]
        x_new[m_start : m_start + self.P] = m_new
        return x_new

    def is_converged(self, x, threshold=1e-6):
        return self.current_residual is not None and self.current_residual < threshold
