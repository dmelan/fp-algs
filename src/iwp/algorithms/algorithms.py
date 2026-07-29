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
    (piecewise-constant) contrast basis with singleton jump groups -- see the
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

    def run(self, x0, max_iterations=1000):
        self.max_iterations = max_iterations
        if self.logger:
            self.logger.info(
                f"Started {self.algo_plot_name} for a maximum of {self.max_iterations} iterations."
            )
        # Preallocate arrays to not count them in memory usage
        self.x_values = np.empty((max_iterations + 1,) + x0.shape, dtype=x0.dtype)
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
            self.x_values[self.iteration] = x
            self.f_values[self.iteration] = self.f(x)
            if self.logger:
                msg = f"Iteration {self.iteration}: f(x) = {self.f_values[self.iteration]:.6f}, time = {time.time() - t0:.3f}s"
                (self.logger.info if self.verbose else self.logger.debug)(msg)
        # Special case for the closed-form algorithm
        if self.iteration == 0:
            x = self.step(x)
            self.iteration += 1
            self.x_values[self.iteration] = x
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
        self.x_values = self.x_values[: self.iteration + 1]
        self.f_values = self.f_values[: self.iteration + 1]
        return x

    def plot_algorithm_convergence(
        self, m, visuals_path, add_marker=False, show=False, save=True
    ):
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

    Block dual step sizes (Eq. (34)). `L` stacks two blocks of very different
    scales -- the PDE coupling `(A u_i - B_i m)_i`, whose norm is driven by
    `||A|| = O(1/h^2)`, and the regularizer block `G m`, with `||G|| = O(1/h)`
    for Total Variation -- so a single dual step `sigma` is throttled by the
    larger one and wastes the freedom of the smaller. Following the diagonal
    preconditioning of Pock and Chambolle (Sec. 4.7, "Choice of the step
    sizes"), `sigma_pde` and `sigma_reg` are the two diagonal entries of the
    block dual metric `Sigma = diag(sigma_pde I, sigma_reg I)`, under which
    the convergence condition becomes

        tau * ||Sigma^(1/2) L||^2 < 1                                (Eq. 34)

    instead of the scalar `tau * sigma * ||L||^2 < 1` (Eq. 31), which is
    recovered by leaving `sigma_reg=None` (it then defaults to `sigma_pde`).
    Note what this does *not* buy (Sec. 4.7): `||Sigma^(1/2) L||` is still
    bounded below by `sqrt(sigma_pde) ||A||`, so the constraint dual stays
    capped at `sigma_pde <~ 1/(tau ||A||^2)` however the metric is balanced --
    the ill-conditioning of the Helmholtz operator sits inside `L` and no
    diagonal rescaling can extract it. Only the regularizer block genuinely
    gains. `iwp.experiments.comparison.block_step_sizes_algorithm3` implements
    the balancing recipe (per-block power iteration, then one primal/dual
    ratio knob `gamma`).

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
        sigma_pde,
        sigma_reg=None,
        prox_dual_reg=None,
        logger=None,
        verbose=False,
    ):
        super().__init__(exp_name, algo_plot_name, f, logger=logger, verbose=verbose)
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
        self.sigma_pde = sigma_pde
        self.sigma_reg = sigma_pde if sigma_reg is None else sigma_reg
        self.prox_dual_reg = prox_dual_reg

        # One-time setup: factor the Woodbury correction matrix (I + tau*C*C^*),
        # shared across all sources since C does not depend on i, cf. Eq. (32).
        J = C.shape[0]
        woodbury_matrix = sp.eye(J, format="csc", dtype=complex) + tau * (
            C @ self.C_star
        )
        self.woodbury_lu = sp.linalg.splu(woodbury_matrix.tocsc())

        Q = G.shape[0] if G is not None else 0
        self.v_pde = [np.zeros(L, dtype=complex) for _ in range(I)]
        self.v_reg = np.zeros(Q, dtype=complex)
        self.u_bar = None
        self.m_bar = None
        self.current_residual = None

    def _prox_data(self, i, v):
        # (C^*C + tau^-1 I)^-1 (C^*d_i + tau^-1 v) via Sherman-Morrison-Woodbury, Eq. (31)-(32)
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

        # PDE dual update (parallelizable over i), at the PDE block's own dual
        # step size sigma_pde of the metric Sigma (Eq. (34)).
        for i in range(self.I):
            self.v_pde[i] = self.v_pde[i] + self.sigma_pde * (
                self.A @ self.u_bar[i] - self.B[i] @ self.m_bar
            )
        # Regularizer dual update (TV or Tikhonov, cf. `prox_dual_reg`), at the
        # regularizer block's own dual step size sigma_reg.
        if self.G is not None:
            self.v_reg = self.prox_dual_reg(
                self.v_reg + self.sigma_reg * (self.G @ self.m_bar), self.sigma_reg
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
      every call -- exactly what the original report's FISTA `prox_J_2`
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
        self.cg_eta0 = cg_eta0
        self.cg_gamma = cg_gamma
        # The residual-to-projection bound (Eq. 53) only needs a summable
        # tolerance schedule, not machine precision; as k grows, eta_k =
        # eta0*gamma^k decays geometrically and would otherwise eventually
        # demand an unreasonably (or, once it underflows, impossibly) tight
        # CG solve every single outer iteration. `cg_min_tol` floors the
        # *relative* tolerance (the report's own Eq. 54 remark: "floored at
        # eta_k >= eps_mach ||r^k||" -- we use a looser, more practical floor
        # by default), and `cg_maxiter` hard-caps inner iterations regardless,
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

    def _capacitance_matvec(self, c):
        # c -> c + sum_i N_i^H N_i c, computed via 2 triangular solves per
        # source without ever forming N_i or S (matrix-free, Sec. 5.4 remark).
        total = c.copy()
        for i in range(self.I):
            t = self.A_lu.solve(self.B[i] @ c)
            h = self.A_lu.solve(t, trans="H")
            total = total + self.B_star[i] @ h
        return total

    def project(self, u_list, m, iteration=None):
        self.n_calls += 1
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
            g_list = []
            for ri in r_list:
                t = self.A_lu.solve(ri)
                g_list.append(self.A_lu.solve(t, trans="H"))
            s = sum(self.B_star[i] @ g_list[i] for i in range(self.I))

            if self.method == "smw":
                c = sla.lu_solve(self.cap_lu_piv, s)
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
                    (self.P, self.P), matvec=self._capacitance_matvec, dtype=complex
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
                w_list.append(g_list[i] - h)

        u_new = [u_list[i] - self.A_star @ w_list[i] for i in range(self.I)]
        m_new = m + sum(self.B_star[i] @ w_list[i] for i in range(self.I))
        return u_new, m_new


class ProjectedChambollePock(FixedPointAlgorithm):
    """Inexact Projected Chambolle-Pock algorithm (Algorithm 5, Sec. 5.5): the
    complementary instantiation to `ChambollePock` (Algorithm 3). Here the
    PDE constraint is kept in the *primal* proximable block `f = iota_C`,
    evaluated by the closed-form (exact or inexact) affine projection of
    Sec. 5.3-5.4 through an `AffineConstraintProjector`, while the
    data-fidelity term and the regularizer on m (TV or Tikhonov) are
    *dualized* (Eq. (43)-(45)) -- the exact reverse of Algorithm 3, where the
    PDE constraint is dualized and the data term is proxed. The key practical
    payoff (Sec. 5.1, 5.6) is that the step-size condition no longer involves
    `||A||`, only `||C||` and the regularizer operator's norm, making it far
    less sensitive to mesh refinement than `ChambollePock`.

    State layout is the same as `ChambollePock`: `x = [u_0, ..., u_{I-1}, m]`.
    `sigma_dat` and `sigma_reg` are the (block-preconditioned, Eq. (56))
    dual step sizes for the data and regularizer blocks respectively; a
    single scalar `sigma` can be passed to both for the non-preconditioned
    variant (Eq. (33)).
    """

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
        elif reg_mode == "none":
            self.prox_reg_dual = None
            Q = 0
        else:
            raise ValueError(f"Unknown reg_mode: {reg_mode!r}")

        self.data_dual_prox = [make_data_dual_prox(d[i]) for i in range(I)]
        self.v_dat = [np.zeros(C.shape[0], dtype=complex) for _ in range(I)]
        self.v_reg = np.zeros(Q, dtype=complex)
        self.x_bar = None
        self.current_residual = None
        self.current_step_norm = None

    def _split(self, x):
        u = [x[i * self.L : (i + 1) * self.L] for i in range(self.I)]
        m = x[self.I * self.L : self.I * self.L + self.P]
        return u, m

    def step(self, x):
        u, m = self._split(x)
        if self.x_bar is None:
            u_bar, m_bar = u, m
        else:
            u_bar, m_bar = self._split(self.x_bar)

        # Dual update of the data term (Eq. (44)), parallelizable over i.
        v_dat_new = [
            self.data_dual_prox[i](
                self.v_dat[i] + self.sigma_dat * (self.C @ u_bar[i]), self.sigma_dat
            )
            for i in range(self.I)
        ]

        # Dual update of the regularizer on m (TV Eq. (45) or Tikhonov).
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

        # Gradient-type step z = x - tau * K^* v (step 4, Algorithm 5).
        z_u = [u[i] - self.tau * (self.C_star @ v_dat_new[i]) for i in range(self.I)]
        if self.reg_mode == "tv":
            z_m = m - self.tau * (self.G_star @ v_reg_new)
        elif self.reg_mode == "tikhonov":
            z_m = m - self.tau * v_reg_new
        else:
            z_m = m.copy()

        # Exact/inexact projection onto the affine PDE constraint (steps 5-7).
        u_new, m_new = self.projector.project(z_u, z_m, iteration=self.iteration)

        x_new = np.empty_like(x)
        for i in range(self.I):
            x_new[i * self.L : (i + 1) * self.L] = u_new[i]
        x_new[self.I * self.L : self.I * self.L + self.P] = m_new

        # Over-relaxation (theta = 1, step 8).
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

    def is_converged(self, x, threshold=1e-6):
        # ChambollePock/DistributedChambollePock track the (asymptotic) PDE
        # feasibility gap as their convergence proxy; here that gap is ~0 at
        # every iteration by construction (Sec. 5.1), so we instead use the
        # fixed-point step size ||x_{k+1} - x_k||, a standard practical
        # stopping rule for primal-dual algorithms without a cheap duality
        # gap (see the notebook's discussion of this choice).
        return self.current_step_norm is not None and self.current_step_norm < threshold


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

    Block dual step sizes. As in `ChambollePock`, the PDE and regularizer
    blocks of `L` carry their own dual step sizes `sigma_pde`/`sigma_reg`
    (block metric `Sigma = diag(sigma_pde I, sigma_reg I)`, Eq. (34)); the
    per-agent regularizer duals `v_reg[s]` all use `sigma_reg`, since the
    consensus step keeps every `m_s` equal and the agents therefore see the
    same regularizer block. Leaving `sigma_reg=None` recovers the scalar
    metric of Eq. (31).
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
        sigma_pde,
        sigma_reg=None,
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
        self.sigma_pde = sigma_pde
        self.sigma_reg = sigma_pde if sigma_reg is None else sigma_reg
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
                self.v_pde[i] = self.v_pde[i] + self.sigma_pde * (
                    self.A @ self.u_bar[i] - self.B[i] @ self.m_bar
                )
                u_new[i] = self._prox_data(
                    i, u[i] - self.tau * (self.A_star @ self.v_pde[i])
                )

            if self.G is not None:
                self.v_reg[s] = self.prox_dual_reg(
                    self.v_reg[s] + self.sigma_reg * (self.G @ self.m_bar),
                    self.sigma_reg,
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
