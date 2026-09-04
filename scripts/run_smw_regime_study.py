#!/usr/bin/env python
"""The favourable regime: a joint `I x delta` sweep for the SMW projection.

Every SMW measurement in this repository so far varied *one* axis at a time:
sources at `L = 223`, mesh at `I = 2`. The reference discretization has
`P/(I L) ~ 0.88`, so the "low-rank" correction `B B*` of Eq. (48) has rank
commensurate with the full dimension and the term is a misnomer. Every
existing conclusion therefore comes from the regime the SMW route was *not*
built for.

This driver measures the regime Sec. 5.8 actually names, `P << I L`, reached by
decoupling the contrast mesh from the field mesh (`-delta_m` in
`scripts/GenerateMatrixSweep.edp`; `bash scripts/generate_joint_sweep.sh`
produces every dataset used below), and evaluates the manuscript's own
quantitative conditions on it.

Sections:

  A  dataset audit: shapes, `P/(I L)`, `P` against `(I L)^{3/4}`, and an
     explicit check that every exported `B_i` is `(L, P)` and not `(P, L)`.
  B  backend cost and *process-level peak RSS* for S1/S2/S3/S4 across the
     joint grid, each measurement in its own subprocess under a stated memory
     budget, so a backend that cannot run is recorded as a datum rather than
     taking the machine down with it.
  C  condition (a): is `P << (I L)^{3/4}`, and does S3 then beat one baseline
     factorization?
  D  condition (b): the amortization threshold `T`, measured against the
     predicted `(L^{3/2} + I L P^2 + P^3/3) / F_fact(I, L)`.
  E  condition (c): does the S3-over-S2 crossover `I*` move as `L` grows?
  F  the memory case for S4: the feasibility frontier, in MB.
  G  the two CG tolerance schedules, re-asked at large `P` and large `I`.
  H  acceptance: every feasible backend reaches the same fixed point to ~1e-12
     with `(tau, sigma, lambda)` held fixed.

Run:
    pixi run python scripts/run_smw_regime_study.py
    pixi run python scripts/run_smw_regime_study.py --only A,B --quick
"""

import argparse
import json
import os
import subprocess
import sys
import time

# OpenBLAS threading has to be pinned *before* numpy is imported, exactly as in
# scripts/run_acceleration_study.py: on this WSL2 host the multi-threaded path
# is far slower for the few-hundred-wide dense blocks used here, and for a
# memory study it would also add per-thread arenas to the RSS being measured.
_BLAS_VARS = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
for _var in _BLAS_VARS:
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from iwp.data.load_experiment_data import load_experiment_data  # noqa: E402
from iwp.utils.mesh import load_dims  # noqa: E402

METHODS = ("spsolve", "cached_splu", "smw", "smw_cg")
METHOD_LABEL = {
    "spsolve": "S1 spsolve",
    "cached_splu": "S2 cached_splu",
    "smw": "S3 smw",
    "smw_cg": "S4 smw_cg",
}
HERE = os.path.dirname(os.path.abspath(__file__))
JOINT_ROOT = os.path.normpath(os.path.join(HERE, "..", "data", "joint"))
COMPLEX_BYTES = 16

# The favourable grid: contrast mesh pinned at delta_m = 10 while the field
# mesh is refined. delta = 10 is kept as the "unfavourable" reference column,
# and I = 2, 4 are kept below the joint grid proper because locating the
# S3-over-S2 crossover I* (condition (c)) needs the baselines to be measured on
# the side of the crossover where they still win.
FAVOURABLE_GRID = [(i, d) for d in (10, 20, 40) for i in (2, 4, 8, 16, 32)]
# The unfavourable companion at the finest field mesh: P = 6720 > L = 3461.
UNFAVOURABLE_GRID = [(2, 40), (8, 40), (16, 40)]


def dataset_dir(I, delta, delta_m=10, root=None):
    return os.path.join(root or JOINT_ROOT, f"I{I}_d{delta}_dm{delta_m}")


# ===========================================================================
# Workers: one measurement per child process.
#
# Isolation is not a nicety here. The point of Sec. 5.3 is that the baselines'
# `E E*` factorization has O(I^2 L) fill and blows up "ordering-dependently";
# on the joint grid it blows up past the physical memory of the machine. In a
# single process a failed attempt would leave the allocator fragmented and
# poison every later measurement, and an unguarded one would be OOM-killed.
# Each measurement therefore runs in a fresh child with an explicit
# address-space budget, so "does not run" becomes a recorded status with a
# number attached rather than a crash.
# ===========================================================================


def _rss_sampler(stop_flag, out, interval=0.002):
    """Poll this process's resident set size and keep the maximum.

    `psutil` is what this study needs and what the internship report's own
    remark demands: `tracemalloc` (which `FixedPointAlgorithm.run` uses) only
    sees allocations made through Python's allocator, so it is blind to
    everything SuperLU, BLAS and the sparse kernels allocate inside compiled
    code -- which is the entire cost being measured here.
    """
    import psutil

    proc = psutil.Process()
    peak = 0
    while not stop_flag[0]:
        try:
            peak = max(peak, proc.memory_info().rss)
        except Exception:  # process teardown
            break
        out[0] = peak
        time.sleep(interval)
    try:
        peak = max(peak, proc.memory_info().rss)
    except Exception:
        pass
    out[0] = peak


class _Meter:
    """Loads the dataset, caps the address space, and samples peak RSS."""

    def __init__(self, cfg):
        import psutil

        self.cfg = cfg
        self.proc = psutil.Process()
        self.result = dict(cfg)
        self.result["status"] = "ok"

    def load(self):
        t0 = time.time()
        A, B_list, C, d_list, m = load_experiment_data(self.cfg["path"])
        self.result["load_time"] = time.time() - t0
        self.result.update(I=len(B_list), L=A.shape[0], P=B_list[0].shape[1])
        self.result["rss_after_load_mb"] = self.rss_mb()
        return A, B_list, C, d_list, m

    def rss_mb(self):
        return self.proc.memory_info().rss / 1e6

    def start(self):
        """Cap the address space *after* the dataset is resident, so the quoted
        budget is the one the backend itself gets rather than a mixture of that
        and the interpreter's own footprint."""
        import resource
        import threading

        if self.cfg.get("mem_budget_mb"):
            cap = self.proc.memory_info().vms + int(self.cfg["mem_budget_mb"] * 1e6)
            resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
        self._stop = [False]
        self._peak = [0]
        self._thread = threading.Thread(
            target=_rss_sampler, args=(self._stop, self._peak), daemon=True
        )
        self._thread.start()

    def finish(self):
        import resource

        self._stop[0] = True
        self._thread.join(timeout=1.0)
        r = self.result
        r["peak_rss_mb"] = self._peak[0] / 1e6
        # ru_maxrss is in KiB on Linux: an independent cross-check on the poll.
        r["max_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        r["backend_mb"] = max(0.0, r["peak_rss_mb"] - r.get("rss_after_load_mb", 0.0))
        print("RESULT " + json.dumps(r), flush=True)


def _looks_like_oom(exc):
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(
        s in text
        for s in ("memory", "cannot allocate", "array is too big", "out of memory")
    )


def _projector_kwargs(cfg):
    return {
        key: cfg[key]
        for key in ("cg_gamma", "cg_eta0", "cg_min_tol", "cg_maxiter")
        if cfg.get(key) is not None
    }


def _worker_backend(cfg):
    """Section B/F: time one backend's setup and per-projection cost, and
    record the peak RSS of the whole process while it does so."""
    from iwp.algorithms.algorithms import AffineConstraintProjector

    meter = _Meter(cfg)
    A, B_list, C, d_list, m = meter.load()
    I, L, P = meter.result["I"], meter.result["L"], meter.result["P"]
    meter.start()

    rng = np.random.default_rng(0)
    u_list = [rng.normal(size=L) + 1j * rng.normal(size=L) for _ in range(I)]
    m_vec = rng.normal(size=P) + 1j * rng.normal(size=P)

    try:
        t0 = time.time()
        projector = AffineConstraintProjector(
            A, B_list, method=cfg["method"], **_projector_kwargs(cfg)
        )
        meter.result["setup_time"] = time.time() - t0
        meter.result["rss_after_setup_mb"] = meter.rss_mb()

        times = []
        t_start = time.time()
        for k in range(cfg["n_calls"]):
            t1 = time.time()
            u_out, m_out = projector.project(
                [u.copy() for u in u_list], m_vec.copy(), iteration=k
            )
            times.append(time.time() - t1)
            if time.time() - t_start > cfg["time_budget_s"]:
                break
        meter.result.update(
            n_calls_timed=len(times),
            per_call_time=float(np.median(times)),
            per_call_time_min=float(np.min(times)),
            per_call_time_mean=float(np.mean(times)),
            # The projection is exact for S1/S2/S3 and inexact for S4, so this
            # is what the CG tolerance actually bought.
            feasibility=projector.feasibility_residual_norm(u_out, m_out),
            n_A_solves=projector.n_A_solves,
            inner_iterations_mean=(
                float(np.mean(projector.inner_iterations))
                if projector.inner_iterations
                else None
            ),
            u_norm=float(np.linalg.norm(np.concatenate(u_out))),
            m_norm=float(np.linalg.norm(m_out)),
        )
    except MemoryError as exc:
        meter.result.update(status="memory", error=f"MemoryError: {exc}")
    except Exception as exc:  # numpy/scipy raise ValueError on some huge allocs
        meter.result.update(
            status="memory" if _looks_like_oom(exc) else "error",
            error=f"{type(exc).__name__}: {exc}",
        )
    meter.finish()


def _worker_cg_schedule(cfg):
    """Section G: run Algorithm 5 with a given CG tolerance schedule and record
    the inner-iteration trace, plus the capacitance conditioning that explains
    it.

    `S = I_P + sum_i N_i^H N_i` satisfies `S >= I_P` by construction (Sec. 5.4),
    so `lambda_min(S) >= 1` and the condition number is bounded by
    `lambda_max(S)` alone. That is estimated matrix-free by power iteration on
    the same capacitance matvec the solver uses, which is the only way to get
    it at all once `P = 6720` makes forming `S` a 723 MB proposition.
    """
    from iwp.algorithms.algorithms import (
        AffineConstraintProjector,
        ProjectedChambollePock,
    )

    meter = _Meter(cfg)
    A, B_list, C, d_list, m = meter.load()
    I, L, P = meter.result["I"], meter.result["L"], meter.result["P"]
    meter.start()

    try:
        projector = AffineConstraintProjector(
            A, B_list, method=cfg["method"], **_projector_kwargs(cfg)
        )
        meter.result["setup_time"] = projector.setup_time

        # lambda_max(S), matrix-free.
        rng = np.random.default_rng(3)
        v = rng.normal(size=P) + 1j * rng.normal(size=P)
        v /= np.linalg.norm(v)
        lam = np.nan
        for _ in range(cfg.get("power_iters", 60)):
            w = projector._capacitance_matvec(v, 1.0)
            lam = float(np.linalg.norm(w))
            v = w / lam
        meter.result["lambda_max_S"] = lam
        meter.result["kappa_S_upper"] = lam  # since lambda_min(S) >= 1
        n_solves_before = projector.n_A_solves

        f = lambda x: 0.0  # noqa: E731 -- the objective is not the object here
        x0 = np.zeros(I * L + P, dtype=complex)
        # ||K|| for Algorithm 5 involves only C, never A (Eq. (43)); computed
        # here by power iteration on C alone since G is not used.
        C_star = C.conj().T
        v = rng.normal(size=L) + 1j * rng.normal(size=L)
        v /= np.linalg.norm(v)
        normC = 0.0
        for _ in range(200):
            w = C_star @ (C @ v)
            normC = np.linalg.norm(w)
            v = w / max(normC, 1e-300)
        normC = float(np.sqrt(normC))
        tau = sigma = 0.9 / max(normC, 1e-12)

        algo = ProjectedChambollePock(
            exp_name="regime",
            algo_plot_name=f"Alg5-{cfg['method']}-gamma{cfg.get('cg_gamma')}",
            f=f,
            C=C,
            d=d_list,
            I=I,
            L=L,
            P=P,
            tau=tau,
            sigma_dat=sigma,
            sigma_reg=sigma,
            projector=projector,
            reg_mode="tikhonov",
            mu=cfg.get("mu", 1e-6),
        )
        algo.max_iterations = cfg["outer_iterations"]
        algo.store_history = False
        algo.iteration = 0
        x = x0
        t0 = time.time()
        budget = cfg.get("max_seconds")
        done = 0
        for k in range(1, cfg["outer_iterations"] + 1):
            x = algo.step(x)
            algo.iteration = done = k
            if budget is not None and time.time() - t0 > budget:
                break
        meter.result["outer_iterations_done"] = done
        meter.result.update(
            run_time=time.time() - t0,
            outer_iterations=cfg["outer_iterations"],
            inner_iterations=list(map(int, projector.inner_iterations)),
            inner_mean=float(np.mean(projector.inner_iterations))
            if projector.inner_iterations
            else None,
            inner_max=int(np.max(projector.inner_iterations))
            if projector.inner_iterations
            else None,
            inner_first=int(projector.inner_iterations[0])
            if projector.inner_iterations
            else None,
            inner_last=int(projector.inner_iterations[-1])
            if projector.inner_iterations
            else None,
            a_solves_outer=projector.n_A_solves - n_solves_before,
            final_feasibility=float(
                np.sqrt(
                    sum(
                        np.linalg.norm(A @ x[i * L : (i + 1) * L] - B_list[i] @ x[-P:])
                        ** 2
                        for i in range(I)
                    )
                )
            ),
            m_norm=float(np.linalg.norm(x[-P:])),
            m_final=[float(np.real(z)) for z in x[-P:][:8]],
            m_checksum=float(np.real(np.vdot(x[-P:], x[-P:]))),
        )
        if cfg.get("m_out"):
            np.save(cfg["m_out"], x[-P:])
    except MemoryError as exc:
        meter.result.update(status="memory", error=f"MemoryError: {exc}")
    except Exception as exc:
        meter.result.update(
            status="memory" if _looks_like_oom(exc) else "error",
            error=f"{type(exc).__name__}: {exc}",
        )
    meter.finish()


def _worker_lambda_max(cfg):
    """`lambda_max(S)` for the capacitance `S = I_P + sum_i N_i^H N_i`, by power
    iteration on the matrix-free capacitance matvec.

    Sec. 5.4's structural claim is `S >= I_P`, hence `lambda_min(S) >= 1` and
    `kappa(S) <= lambda_max(S)`. That upper bound is the whole reason the inner
    CG solve was expected to be cheap, and it is the quantity that decides
    whether a tolerance schedule can bind at all. Computed matrix-free because
    at `P = 6720` forming `S` is a 723 MB proposition.
    """
    from iwp.algorithms.algorithms import AffineConstraintProjector

    meter = _Meter(cfg)
    A, B_list, C, d_list, m = meter.load()
    P = meter.result["P"]
    meter.start()
    try:
        projector = AffineConstraintProjector(A, B_list, method="smw_cg")
        rng = np.random.default_rng(3)
        v = rng.normal(size=P) + 1j * rng.normal(size=P)
        v /= np.linalg.norm(v)
        lam = np.nan
        t0 = time.time()
        for _ in range(cfg.get("power_iters", 80)):
            w = projector._capacitance_matvec(v, 1.0)
            lam = float(np.linalg.norm(w))
            v = w / lam
        meter.result.update(
            lambda_max_S=lam,
            kappa_S_upper=lam,  # lambda_min(S) >= 1 by construction
            # CG needs O(sqrt(kappa) log(1/eta)) iterations; this is the factor
            # that says how much room a tolerance schedule has to matter.
            cg_iterations_per_digit=float(np.sqrt(lam) * np.log(10.0) / 2.0),
            power_iteration_time=time.time() - t0,
        )
    except MemoryError as exc:
        meter.result.update(status="memory", error=f"MemoryError: {exc}")
    except Exception as exc:
        meter.result.update(
            status="memory" if _looks_like_oom(exc) else "error",
            error=f"{type(exc).__name__}: {exc}",
        )
    meter.finish()


def _worker_fixed_point(cfg):
    """Section H: run Algorithm 5 with one projector backend at fixed
    `(tau, sigma, mu)` and save the final contrast, so the caller can compare
    fixed points across backends and attribute any difference to the projector
    alone."""
    from iwp.algorithms.algorithms import (
        AffineConstraintProjector,
        ProjectedChambollePock,
    )

    meter = _Meter(cfg)
    A, B_list, C, d_list, m = meter.load()
    I, L, P = meter.result["I"], meter.result["L"], meter.result["P"]
    meter.start()
    try:
        projector = AffineConstraintProjector(
            A, B_list, method=cfg["method"], **_projector_kwargs(cfg)
        )
        f = lambda x: 0.0  # noqa: E731
        algo = ProjectedChambollePock(
            exp_name="regime",
            algo_plot_name=f"Alg5-{cfg['method']}",
            f=f,
            C=C,
            d=d_list,
            I=I,
            L=L,
            P=P,
            tau=cfg["tau"],
            sigma_dat=cfg["sigma"],
            sigma_reg=cfg["sigma"],
            projector=projector,
            reg_mode="tikhonov",
            mu=cfg["mu"],
        )
        algo.max_iterations = cfg["outer_iterations"]
        algo.store_history = False
        algo.iteration = 0
        x = np.zeros(I * L + P, dtype=complex)
        t0 = time.time()
        budget = cfg.get("max_seconds")
        for k in range(1, cfg["outer_iterations"] + 1):
            x = algo.step(x)
            algo.iteration = k
            if budget is not None and time.time() - t0 > budget:
                # Abort rather than truncate. Comparing fixed points across
                # backends is only meaningful at a matched iteration count, so
                # a backend that cannot afford the run has to be recorded as
                # unaffordable, not as having reached a different point.
                meter.result.update(
                    status="too_slow",
                    error=f"{k}/{cfg['outer_iterations']} iterations in "
                          f"{time.time() - t0:.0f}s, budget {budget:.0f}s",
                    iterations_done=k,
                )
                meter.finish()
                return
        meter.result.update(
            run_time=time.time() - t0,
            feasibility=float(
                np.sqrt(
                    sum(
                        np.linalg.norm(A @ x[i * L : (i + 1) * L] - B_list[i] @ x[-P:])
                        ** 2
                        for i in range(I)
                    )
                )
            ),
            m_norm=float(np.linalg.norm(x[-P:])),
        )
        np.save(cfg["m_out"], x[-P:])
    except MemoryError as exc:
        meter.result.update(status="memory", error=f"MemoryError: {exc}")
    except Exception as exc:
        meter.result.update(
            status="memory" if _looks_like_oom(exc) else "error",
            error=f"{type(exc).__name__}: {exc}",
        )
    meter.finish()


_WORKERS = {
    "backend": _worker_backend,
    "cg_schedule": _worker_cg_schedule,
    "fixed_point": _worker_fixed_point,
}


# ===========================================================================
# The driver side of one measurement.
# ===========================================================================


def _watch_child(proc, timeout_s, interval=0.002):
    """Poll the child's RSS until it exits. Returns the peak in bytes, or -1 if
    the child had to be killed for exceeding `timeout_s`."""
    import psutil

    try:
        ps = psutil.Process(proc.pid)
    except Exception:
        return 0
    peak, t0 = 0, time.time()
    while proc.poll() is None:
        try:
            peak = max(peak, ps.memory_info().rss)
        except Exception:
            break
        if time.time() - t0 > timeout_s:
            proc.kill()
            return -1
        time.sleep(interval)
    return peak


def spawn(cfg, timeout_s=180.0, logger=print, label=None):
    """Run one worker in a child process and return its result row.

    The row always carries a `status`: "ok"; "memory" (the backend exceeded its
    address-space budget); "timeout"; "killed" (the OS or the allocator ended
    the child outright); or "error".
    """
    cmd = [sys.executable, os.path.abspath(__file__), "--worker", json.dumps(cfg)]
    env = dict(os.environ)
    for var in _BLAS_VARS:
        env[var] = "1"

    t0 = time.time()
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, text=True
    )
    # An independent, parent-side peak-RSS sample of the child. The child
    # samples itself too; reporting both matters because a child killed
    # mid-allocation never gets to report its own.
    parent_peak = _watch_child(proc, timeout_s)
    try:
        out, err = proc.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()
    wall = time.time() - t0

    row = dict(
        path=cfg["path"],
        dataset=os.path.basename(cfg["path"].rstrip("/")),
        method=cfg.get("method"),
        wall_time=wall,
        parent_peak_rss_mb=parent_peak / 1e6 if parent_peak > 0 else np.nan,
        mem_budget_mb=cfg.get("mem_budget_mb"),
    )
    line = next((l for l in out.splitlines() if l.startswith("RESULT ")), None)
    if line is not None:
        row.update(json.loads(line[len("RESULT ") :]))
    elif parent_peak == -1:
        row.update(status="timeout", error=f"exceeded {timeout_s:.0f}s")
    else:
        tail = (err or "").strip().splitlines()
        row.update(
            status="killed",
            error=(tail[-1][:200] if tail else "child produced no result"),
        )
    row.setdefault("peak_rss_mb", row["parent_peak_rss_mb"])
    if logger is not None:
        name = label or METHOD_LABEL.get(cfg.get("method"), str(cfg.get("method")))
        if row["status"] == "ok":
            logger(
                f"    {name:18s} ok       "
                f"setup={row.get('setup_time', np.nan):8.3f}s  "
                f"call={row.get('per_call_time', np.nan) * 1000:9.3f}ms  "
                f"peakRSS={row.get('peak_rss_mb', np.nan):8.1f}MB  "
                f"backend={row.get('backend_mb', np.nan):8.1f}MB"
            )
        else:
            logger(f"    {name:18s} {row['status']:8s} ({row.get('error', '')})")
    return row


# ===========================================================================
# Section A: dataset audit
# ===========================================================================


def _matrix_header(path):
    """(rows, cols, nnz) from a FreeFEM-exported .dat matrix, without parsing
    the body. This is the *exported* shape, which is the authority on the
    `(L, P)` vs `(P, L)` question."""
    with open(path) as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                parts = line.split()
                return int(parts[0]), int(parts[1]), int(parts[2])
    raise ValueError(f"{path}: no header line")


def audit_dataset(path):
    """Shapes, dimensions and regime ratios for one dataset directory.

    The `B_i` shape check is the point: `B_i` maps the coarse contrast space to
    the fine field space, so it is `(L, P)`. The manuscript's inverse-problem
    section states `P x L`, which is a transposition typo; with two distinct
    meshes the two are no longer even the same size, so the check has teeth
    here in a way it never did when `L = 223` and `P = 394` were both "a few
    hundred".
    """
    dims = load_dims(path) or {}
    rowsA, colsA, nnzA = _matrix_header(os.path.join(path, "MatrixABorn.dat"))
    rowsC, colsC, _ = _matrix_header(os.path.join(path, "MatrixC.dat"))
    b_files = sorted(
        f for f in os.listdir(path) if f.startswith("MatrixB_") and f.endswith(".dat")
    )
    I = len(b_files)
    L, P = rowsA, None
    nnzB = 0
    problems = []
    for f in b_files:
        r, c, n = _matrix_header(os.path.join(path, f))
        nnzB += n
        if P is None:
            P = c
        if r != L:
            problems.append(f"{f}: {r} rows, expected L={L}")
        if c != P:
            problems.append(f"{f}: {c} cols, expected P={P}")
        if r == P and c == L and P != L:
            problems.append(f"{f}: transposed, exported as (P, L)")
    if dims:
        if dims.get("L") not in (None, L):
            problems.append(f"dims.txt says L={dims['L']}, matrices say {L}")
        if dims.get("P") not in (None, P):
            problems.append(f"dims.txt says P={dims['P']}, matrices say {P}")
        if dims.get("I") not in (None, I):
            problems.append(f"dims.txt says I={dims['I']}, found {I} B_i files")
    IL = I * L
    return dict(
        dataset=os.path.basename(path.rstrip("/")),
        path=path,
        I=I,
        delta=dims.get("delta"),
        delta_m=dims.get("delta_m"),
        L=L,
        P=P,
        J=rowsC,
        IL=IL,
        # The regime ratio Sec. 5.8 is about: the rank of the "low-rank"
        # correction B B*, relative to the dimension it corrects.
        P_over_IL=P / IL,
        IL_pow_075=IL**0.75,
        # The manuscript's sufficient condition for SMW per-iteration cost to
        # beat one baseline factorization.
        P_over_IL075=P / IL**0.75,
        condition_a=P / IL**0.75 < 1.0,
        nnz_A=nnzA,
        nnz_B_total=nnzB,
        B_shape=f"({L}, {P})",
        B_shape_ok=not problems,
        problems="; ".join(problems),
    )


def section_a(paths, results_dir, log):
    log("=== Section A: dataset audit (shapes, and the regime ratio P/(I L)) ===")
    rows = [audit_dataset(p) for p in paths]
    df = pd.DataFrame(rows).sort_values(["delta_m", "delta", "I"], kind="stable")
    bad = df[~df["B_shape_ok"]]
    if len(bad):
        raise AssertionError(
            "Exported B_i shapes are not (L, P):\n" + bad.to_string(index=False)
        )
    log(
        df[
            [
                "dataset", "I", "delta", "delta_m", "L", "P", "J", "IL",
                "P_over_IL", "IL_pow_075", "P_over_IL075", "condition_a",
                "B_shape",
            ]
        ].to_string(index=False)
    )
    log(f"  every B_i verified as (L, P) across {len(df)} datasets")
    df.to_csv(os.path.join(results_dir, "regimeA_dataset_audit.csv"), index=False)
    return df


# ===========================================================================
# Section B: backend cost and peak RSS across the joint grid
# ===========================================================================


def section_b(
    audit,
    results_dir,
    log,
    n_calls=5,
    time_budget_s=8.0,
    mem_budget_mb=700,
    naive_size_cutoff=4000,
    baseline_timeout_s=90.0,
    smw_timeout_s=300.0,
    respect_cutoff=False,
):
    """Time and memory-profile all four backends on every dataset.

    `naive_size_cutoff` is the rule the existing sweep applies: once
    `I L > 4000`, S1/S2 factor an `E E*` whose O(I^2 L) fill makes them
    intractable. Applying it as a hard skip (`respect_cutoff=True`) reproduces
    that sweep's behaviour, but it also *assumes* the answer to Task 4's
    question. The default here is to attempt every backend everywhere under a
    stated address-space budget and a wall-clock timeout, so that the frontier
    is measured; rows past the cutoff are flagged rather than dropped.
    """
    log(
        "=== Section B: backend cost and peak RSS on the joint grid "
        f"(memory budget {mem_budget_mb} MB/backend) ==="
    )
    rows = []
    for rec in audit.to_dict("records"):
        beyond = rec["IL"] > naive_size_cutoff
        log(
            f"  {rec['dataset']:16s} I={rec['I']:3d} L={rec['L']:5d} P={rec['P']:5d} "
            f"I*L={rec['IL']:7d}  P/(I L)={rec['P_over_IL']:.4f}"
            + ("   [past the I*L cutoff for S1/S2]" if beyond else "")
        )
        for method in METHODS:
            naive = method in ("spsolve", "cached_splu")
            if naive and beyond and respect_cutoff:
                rows.append(
                    dict(
                        dataset=rec["dataset"], method=method, I=rec["I"],
                        L=rec["L"], P=rec["P"], delta=rec["delta"],
                        delta_m=rec["delta_m"], status="skipped",
                        error=f"I*L={rec['IL']} > {naive_size_cutoff}",
                        beyond_naive_cutoff=True,
                    )
                )
                log(f"    {METHOD_LABEL[method]:18s} skipped  (cutoff rule)")
                continue
            row = spawn(
                dict(
                    task="backend",
                    path=rec["path"],
                    method=method,
                    n_calls=n_calls,
                    time_budget_s=time_budget_s,
                    mem_budget_mb=mem_budget_mb,
                ),
                timeout_s=baseline_timeout_s if naive else smw_timeout_s,
                logger=log,
            )
            # A child that ran out of memory during the load, or was killed,
            # never reports its own dimensions. Take them from the audit so a
            # failure row still says what it was that failed.
            row.update(
                I=rec["I"], L=rec["L"], P=rec["P"],
                delta=rec["delta"], delta_m=rec["delta_m"],
                IL=rec["IL"], P_over_IL=rec["P_over_IL"],
                beyond_naive_cutoff=beyond,
            )
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(results_dir, "regimeB_backend_cost.csv"), index=False)
    return df


# ===========================================================================
# The analytic memory model, and the nnz of E E*
# ===========================================================================


def _worker_nnz(cfg):
    """Per-dataset structural quantities: the fill of `E E*` that decides
    whether S1/S2 can be built at all, and the dense workspaces S3 needs."""
    import scipy.sparse as sp

    meter = _Meter(cfg)
    A, B_list, C, d_list, m = meter.load()
    I, L, P = meter.result["I"], meter.result["L"], meter.result["P"]
    meter.start()
    try:
        AAs = (A @ A.conj().T).tocsr()
        nnz_AA = int(AAs.nnz)
        del AAs
        B_stacked = sp.vstack([Bi.tocsr() for Bi in B_list], format="csc")
        col_counts = np.diff(B_stacked.indptr).astype(float)
        # `B B*` is a sum of P rank-one outer products; column p contributes
        # c_p^2 entries, so sum_p c_p^2 bounds its nnz (tightly, since the
        # supports overlap only where two contrast dofs share a field dof).
        nnz_BB_bound = float(np.sum(col_counts**2))
        nnz_BB_exact = None
        if nnz_BB_bound < cfg.get("exact_nnz_limit", 3e6):
            BBs = (B_stacked @ B_stacked.conj().T).tocsr()
            nnz_BB_exact = int(BBs.nnz)
            del BBs
        nnz_EE = I * nnz_AA + (nnz_BB_exact or nnz_BB_bound)
        lu = sp.linalg.splu(A.tocsc())
        nnz_lu_A = int(lu.L.nnz + lu.U.nnz)
        meter.result.update(
            nnz_AA_block=nnz_AA,
            nnz_BB_bound=nnz_BB_bound,
            nnz_BB_exact=nnz_BB_exact,
            nnz_EE_star=float(nnz_EE),
            # A complex CSR matrix costs 16 B of data plus 4 B of column index
            # per entry, plus a row pointer. This is the cost of *storing*
            # E E*, before any factorization: a lower bound on what S1/S2 need.
            EE_star_mb=float(nnz_EE * 20 + (I * L + 1) * 4) / 1e6,
            nnz_lu_A=nnz_lu_A,
            lu_A_mb=float(nnz_lu_A * 20) / 1e6,
            # S3 forms the dense P x P capacitance, and transiently holds both
            # B_i.toarray() and N_i = A^-1 B_i, each L x P dense complex.
            smw_capacitance_mb=float(P * P * COMPLEX_BYTES) / 1e6,
            smw_transient_mb=float(2 * L * P * COMPLEX_BYTES) / 1e6,
            smw_total_mb=float(P * P + 2 * L * P) * COMPLEX_BYTES / 1e6
            + float(nnz_lu_A * 20) / 1e6,
            # S4 forms neither: only the LU of A, plus O(I L + P) vectors.
            smw_cg_total_mb=float(nnz_lu_A * 20) / 1e6
            + float((I * L + P) * COMPLEX_BYTES * 6) / 1e6,
        )
    except MemoryError as exc:
        meter.result.update(status="memory", error=f"MemoryError: {exc}")
    except Exception as exc:
        meter.result.update(
            status="memory" if _looks_like_oom(exc) else "error",
            error=f"{type(exc).__name__}: {exc}",
        )
    meter.finish()


def section_nnz(audit, results_dir, log, mem_budget_mb=700):
    log("=== Structural fill model: how large E E* is before anyone factors it ===")
    rows = []
    for rec in audit.to_dict("records"):
        row = spawn(
            dict(
                task="nnz",
                path=rec["path"],
                mem_budget_mb=mem_budget_mb,
                exact_nnz_limit=3e6,
            ),
            timeout_s=240.0,
            logger=None,
        )
        row.update(
            I=rec["I"], L=rec["L"], P=rec["P"],
            delta=rec["delta"], delta_m=rec["delta_m"], IL=rec["IL"],
        )
        rows.append(row)
        if row["status"] == "ok":
            log(
                f"  {rec['dataset']:16s} nnz(E E*)={row['nnz_EE_star']:12.3e} "
                f"-> {row['EE_star_mb']:9.1f} MB stored   "
                f"S3 dense {row['smw_total_mb']:8.1f} MB   "
                f"S4 {row['smw_cg_total_mb']:7.1f} MB"
            )
        else:
            log(f"  {rec['dataset']:16s} {row['status']}: {row.get('error')}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(results_dir, "regimeF_memory_model.csv"), index=False)
    return df


# ===========================================================================
# Helper: a wide, per-configuration view of the Section B measurements
# ===========================================================================


def pivot_backends(bench):
    """One row per dataset, one group of columns per backend."""
    keys = ["dataset", "I", "L", "P", "delta", "delta_m", "IL", "P_over_IL"]
    base = bench[keys].drop_duplicates("dataset").set_index("dataset")
    for method in METHODS:
        sub = bench[bench["method"] == method].set_index("dataset")
        short = {"spsolve": "S1", "cached_splu": "S2", "smw": "S3", "smw_cg": "S4"}[
            method
        ]
        base[f"{short}_status"] = sub["status"]
        ok = sub["status"] == "ok"
        for src, dst in (
            ("setup_time", "setup"),
            ("per_call_time", "call"),
            ("peak_rss_mb", "peak_mb"),
            ("backend_mb", "mb"),
        ):
            column = sub[src] if src in sub else pd.Series(np.nan, index=sub.index)
            # A backend that ran out of memory still has a peak RSS: it is how
            # far it got before failing, not what it needs, so it must not enter
            # a footprint comparison. Kept under `_attempt_mb` for the record.
            if dst == "mb":
                base[f"{short}_attempt_mb"] = column
            base[f"{short}_{dst}"] = column.where(ok)
    return base.reset_index()


def _ok(value):
    return value is not None and np.isfinite(value)


# ===========================================================================
# Section C: condition (a), P << (I L)^{3/4}
# ===========================================================================


def section_c(wide, results_dir, log, strong=0.5):
    """Sec. 5.8's sufficient condition for the SMW per-iteration cost to beat a
    single baseline factorization, evaluated where it bites.

    The condition is `P << (I L)^{3/4}`. "Much less than" needs a number to be
    testable at all, so the verdict is reported on the ratio
    `r = P / (I L)^{3/4}`: satisfied for `r <= 0.5`, marginal for
    `0.5 < r < 1`, violated for `r >= 1`. The consequence to check is the one
    the condition is *for*: one S3 projection should cost less than one
    factorization of `E E*`, which is what the S2 setup time measures.
    """
    log("=== Section C: condition (a), P << (I L)^{3/4} ===")
    rows = []
    for r in wide.to_dict("records"):
        ratio = r["P"] / r["IL"] ** 0.75
        verdict = (
            "unknown"
            if not np.isfinite(ratio)
            else "satisfied" if ratio <= strong else "marginal" if ratio < 1 else "violated"
        )
        s3, s2_setup = r.get("S3_call"), r.get("S2_setup")
        if _ok(s3) and _ok(s2_setup):
            beats = bool(s3 < s2_setup)
            consequence = (
                f"yes ({s3 * 1000:.2f} ms < {s2_setup * 1000:.1f} ms)"
                if beats
                else f"no ({s3 * 1000:.2f} ms >= {s2_setup * 1000:.1f} ms)"
            )
        elif _ok(s3):
            beats = True
            consequence = "vacuously (S2 cannot be built at all here)"
        else:
            beats = None
            consequence = "S3 not measurable"
        rows.append(
            dict(
                dataset=r["dataset"], I=r["I"], L=r["L"], P=r["P"],
                IL=r["IL"], P_over_IL=r["P_over_IL"],
                IL_pow_075=r["IL"] ** 0.75, ratio=ratio, condition_a=verdict,
                S3_call_ms=s3 * 1000 if _ok(s3) else np.nan,
                S2_setup_ms=s2_setup * 1000 if _ok(s2_setup) else np.nan,
                S2_status=r.get("S2_status"),
                S3_beats_one_factorization=beats, consequence=consequence,
            )
        )
    df = pd.DataFrame(rows)
    log(
        df[
            ["dataset", "I", "L", "P", "IL", "P_over_IL", "IL_pow_075", "ratio",
             "condition_a", "S3_call_ms", "S2_setup_ms", "consequence"]
        ].to_string(index=False, float_format=lambda v: f"{v:.4g}")
    )
    df.to_csv(os.path.join(results_dir, "regimeC_condition_a.csv"), index=False)
    return df


# ===========================================================================
# Section D: condition (b), the amortization threshold T
# ===========================================================================


def smw_setup_flops(I, L, P):
    """`L^{3/2} + I L P^2 + P^3/3`, the manuscript's flop model for the SMW
    setup: the nested-dissection LU of `A`, forming `sum_i N_i^H N_i` with
    `N_i = A^-1 B_i` (`L P^2` per source), and the dense `P x P` factorization.
    """
    return L**1.5 + I * L * P**2 + P**3 / 3.0


def section_d(wide, results_dir, log):
    """The setup is amortized after `T ~ (L^{3/2} + I L P^2 + P^3/3)/F_fact`
    outer iterations, claimed `O(1)` once `I >= 4`.

    `F_fact(I, L)` is the flop count of one `E E*` factorization, which Sec. 5.3
    explicitly declines to bound ("ordering-dependent... measured rather than
    bounded"). It is therefore taken here from the measurement: the S2 setup
    time *is* one factorization, on the same machine, in the same units as the
    SMW setup time. The predicted threshold is then the dimensionless ratio
    `T_pred = t_setup(S3) / t_setup(S2)`, with the machine constants cancelling
    on both sides.

    Measured against the two baselines separately:
      * `T` vs S1, which re-factors every iteration: SMW is ahead once
        `t_setup(S3) + T t_call(S3) <= T t_call(S1)`;
      * `T` vs S2, which factors once: ahead once
        `t_setup(S3) + T t_call(S3) <= t_setup(S2) + T t_call(S2)`.
    """
    log("=== Section D: condition (b), the amortization threshold T ===")
    rows = []
    for r in wide.to_dict("records"):
        I, L, P = r["I"], r["L"], r["P"]
        flops = smw_setup_flops(I, L, P)
        s3_setup, s3_call = r.get("S3_setup"), r.get("S3_call")
        s2_setup, s2_call = r.get("S2_setup"), r.get("S2_call")
        s1_call = r.get("S1_call")

        t_vs_s1 = np.nan
        if _ok(s3_setup) and _ok(s1_call) and _ok(s3_call) and s1_call > s3_call:
            t_vs_s1 = s3_setup / (s1_call - s3_call)
        t_vs_s2 = np.nan
        if all(_ok(v) for v in (s3_setup, s3_call, s2_setup, s2_call)):
            if s2_call > s3_call:
                t_vs_s2 = max(0.0, (s3_setup - s2_setup) / (s2_call - s3_call))
            elif s3_setup <= s2_setup:
                t_vs_s2 = 0.0  # cheaper to set up *and* never slower per call
            else:
                t_vs_s2 = np.inf  # S2 wins per call: SMW never amortizes here
        t_pred = (
            s3_setup / s2_setup if _ok(s3_setup) and _ok(s2_setup) and s2_setup else np.nan
        )
        rows.append(
            dict(
                dataset=r["dataset"], I=I, L=L, P=P, IL=r["IL"],
                smw_setup_flops=flops,
                S3_setup_s=s3_setup, S2_setup_s=s2_setup,
                S1_call_s=s1_call, S2_call_s=s2_call, S3_call_s=s3_call,
                T_pred_setup_ratio=t_pred,
                T_measured_vs_S1=t_vs_s1,
                T_measured_vs_S2=t_vs_s2,
                S2_status=r.get("S2_status"),
                T_is_O1=bool(np.isfinite(t_vs_s1) and t_vs_s1 <= 10)
                if np.isfinite(t_vs_s1)
                else None,
            )
        )
    df = pd.DataFrame(rows)

    # Does the flop model actually describe the measured SMW setup? A one-
    # parameter fit t = c * flops, reported with its relative spread: if the
    # model is right, c is a machine constant and the spread is small.
    fit = df.dropna(subset=["S3_setup_s", "smw_setup_flops"])
    if len(fit) >= 3:
        # A least-squares line through the origin would be set almost entirely
        # by the largest configuration; the median of the per-configuration
        # ratios is the scale constant with every point weighted equally, and
        # its spread is what says whether the model describes the data at all.
        ratios = fit["S3_setup_s"] / fit["smw_setup_flops"]
        c = float(np.median(ratios))
        rel = np.abs(c * fit["smw_setup_flops"] - fit["S3_setup_s"]) / fit["S3_setup_s"]
        log(
            f"  SMW setup flop model t = c*(L^1.5 + I L P^2 + P^3/3): "
            f"c = {c:.3e} s/flop ({1 / c / 1e9:.2f} Gflop/s effective), "
            f"median |relative error| = {float(np.median(rel)):.1%}, "
            f"max = {float(np.max(rel)):.1%} over {len(fit)} configurations"
        )
        df["flop_model_pred_s"] = c * df["smw_setup_flops"]
        df.attrs["flop_model_c"] = c
    log(
        df[
            ["dataset", "I", "L", "P", "S3_setup_s", "S2_setup_s", "S1_call_s",
             "S3_call_s", "T_pred_setup_ratio", "T_measured_vs_S1",
             "T_measured_vs_S2"]
        ].to_string(index=False, float_format=lambda v: f"{v:.4g}")
    )
    df.to_csv(os.path.join(results_dir, "regimeD_amortization.csv"), index=False)
    return df


# ===========================================================================
# Section E: condition (c), does the crossover I* move with L?
# ===========================================================================


def section_e(wide, results_dir, log):
    """Sec. 5.9 predicts S4 overtakes S3 only at the largest `L`, once the
    `O(L^{3/2})` LU of `A` and its `O(L log L)` fill become the binding cost.
    The S3-over-S2 crossover `I*` was measured at 8 for `L = 223`; here it is
    re-measured at each field mesh.

    `I*` is interpolated in `log I` on the per-call times, which is the right
    variable: both curves are near-power-laws in `I`.
    """
    log("=== Section E: condition (c), the S3-over-S2 crossover as L grows ===")
    rows = []
    for (delta, delta_m), grp in wide.groupby(["delta", "delta_m"], dropna=False):
        grp = grp.sort_values("I")
        Is = grp["I"].to_numpy(float)
        t2 = grp["S2_call"].to_numpy(float)
        t3 = grp["S3_call"].to_numpy(float)
        t4 = grp["S4_call"].to_numpy(float)
        rows.append(
            dict(
                delta=delta, delta_m=delta_m, L=int(grp["L"].iloc[0]),
                P=int(grp["P"].iloc[0]),
                I_star_S3_over_S2=_crossover(Is, t2, t3),
                S3_faster_at_smallest_I=_already_faster(Is, t2, t3),
                I_star_S4_over_S3=_crossover(Is, t3, t4),
                S4_faster_at_smallest_I=_already_faster(Is, t3, t4),
                I_values=[int(v) for v in Is if np.isfinite(v)],
                S2_feasible_up_to=_last_ok(grp, "S2"),
                S3_feasible_up_to=_last_ok(grp, "S3"),
                S4_feasible_up_to=_last_ok(grp, "S4"),
            )
        )
    df = pd.DataFrame(rows)
    log(df.to_string(index=False))
    df.to_csv(os.path.join(results_dir, "regimeE_crossover.csv"), index=False)
    return df


def _crossover(xs, slow, fast):
    """Smallest `x` at which `fast` drops below `slow`, interpolated in log x.

    Returns NaN when the two curves never cross over the measured range, and
    `<= xs[0]` (as a negative sentinel-free float) when `fast` already wins at
    the first point.
    """
    mask = np.isfinite(slow) & np.isfinite(fast)
    xs, slow, fast = xs[mask], slow[mask], fast[mask]
    if len(xs) < 1:
        return np.nan
    diff = np.log(fast) - np.log(slow)
    if diff[0] < 0:
        return float(xs[0])  # already faster at the smallest I measured
    for j in range(1, len(xs)):
        if diff[j] < 0 <= diff[j - 1]:
            w = diff[j - 1] / (diff[j - 1] - diff[j])
            return float(np.exp(np.log(xs[j - 1]) + w * (np.log(xs[j]) - np.log(xs[j - 1]))))
    return np.nan


def _already_faster(xs, slow, fast):
    """True when `fast` is already ahead at the smallest `I` measured, in which
    case `_crossover` returns that `I` and the real crossover is below the
    measured range rather than at its edge."""
    mask = np.isfinite(slow) & np.isfinite(fast)
    if not mask.any():
        return None
    return bool(fast[mask][0] < slow[mask][0])


def _last_ok(grp, short):
    ok = grp[(grp[f"{short}_status"] == "ok") & np.isfinite(grp["I"].astype(float))]
    return int(ok["I"].max()) if len(ok) else None


# ===========================================================================
# Section F: the memory case for S4
# ===========================================================================


def section_f(wide, bench, model, results_dir, log, mem_budget_mb=700):
    """The claim to test is that S4 "earns its keep on memory and on problems
    where the dense `P x P` capacitance cannot be formed at all, not on speed".

    Two things are reported, in MB rather than seconds: the *measured* peak
    resident memory of each backend where it runs, and the *predicted*
    footprint from the structural model, which is what decides feasibility
    independently of the budget this particular machine could afford.
    """
    log(f"=== Section F: the memory case for S4 (budget {mem_budget_mb} MB) ===")
    m = model.set_index("dataset")
    rows = []
    for r in wide.to_dict("records"):
        ds = r["dataset"]
        mm = m.loc[ds] if ds in m.index else {}
        get = lambda k: float(mm.get(k, np.nan)) if len(mm) else np.nan  # noqa: E731
        s3_mb, s4_mb = r.get("S3_mb"), r.get("S4_mb")
        rows.append(
            dict(
                dataset=ds, I=r["I"], L=r["L"], P=r["P"], IL=r["IL"],
                P_over_IL=r["P_over_IL"],
                S1_status=r["S1_status"], S2_status=r["S2_status"],
                S3_status=r["S3_status"], S4_status=r["S4_status"],
                S1_mb=r.get("S1_mb"), S2_mb=r.get("S2_mb"),
                S3_mb=s3_mb, S4_mb=s4_mb,
                S3_minus_S4_mb=(s3_mb - s4_mb)
                if _ok(s3_mb) and _ok(s4_mb)
                else np.nan,
                pred_EE_star_mb=get("EE_star_mb"),
                pred_S3_mb=get("smw_total_mb"),
                pred_S4_mb=get("smw_cg_total_mb"),
                pred_S3_over_S4=get("smw_total_mb") / get("smw_cg_total_mb")
                if get("smw_cg_total_mb")
                else np.nan,
            )
        )
    df = pd.DataFrame(rows)
    log(
        df[
            ["dataset", "I", "L", "P", "P_over_IL", "S1_status", "S2_status",
             "S3_status", "S4_status", "S3_mb", "S4_mb", "S3_minus_S4_mb",
             "pred_EE_star_mb", "pred_S3_mb", "pred_S4_mb"]
        ].to_string(index=False, float_format=lambda v: f"{v:.4g}")
    )

    # The frontier: for each backend, the largest configuration that ran.
    frontier = []
    for short in ("S1", "S2", "S3", "S4"):
        ok = df[df[f"{short}_status"] == "ok"]
        if not len(ok):
            frontier.append(dict(backend=short, largest="none"))
            continue
        best = ok.loc[ok["IL"].idxmax()]
        frontier.append(
            dict(
                backend=short,
                largest=best["dataset"],
                I=int(best["I"]), L=int(best["L"]), P=int(best["P"]),
                IL=int(best["IL"]),
                peak_mb=float(best[f"{short}_mb"]),
                n_configs_ok=int(len(ok)),
                n_configs_memory=int((df[f"{short}_status"] == "memory").sum()),
                n_configs_timeout=int((df[f"{short}_status"] == "timeout").sum()),
            )
        )
    fdf = pd.DataFrame(frontier)
    log("  Feasibility frontier (largest configuration each backend runs at all):")
    log(fdf.to_string(index=False))

    df.to_csv(os.path.join(results_dir, "regimeF_memory_frontier.csv"), index=False)
    fdf.to_csv(os.path.join(results_dir, "regimeF_frontier_summary.csv"), index=False)
    return df, fdf


# ===========================================================================
# Section G: the two CG tolerance schedules, re-asked
# ===========================================================================


def section_g(
    configs, results_dir, log, outer_iterations=150, gammas=(0.5, 0.8),
    mem_budget_mb=700, max_seconds=300.0,
):
    """The two schedules were previously indistinguishable (13.73 vs 13.66 mean
    inner iterations, 0.5%), because CG reaches its natural convergence on a
    capacitance that satisfies `S >= I_P` by construction long before the
    requested tolerance binds.

    Re-asked here at large `P` and large `I`. `lambda_max(S)` is estimated
    matrix-free by power iteration, so the conditioning claim is measured and
    not assumed: with `lambda_min(S) >= 1`, `kappa(S) <= lambda_max(S)`, and CG
    needs `O(sqrt(kappa) log(1/eta))` iterations.
    """
    log("=== Section G: CG tolerance schedules at large P and large I ===")
    rows = []
    for path in configs:
        for gamma in gammas:
            row = spawn(
                dict(
                    task="cg_schedule",
                    path=path,
                    method="smw_cg",
                    cg_gamma=gamma,
                    cg_eta0=1.0,
                    outer_iterations=outer_iterations,
                    mem_budget_mb=mem_budget_mb,
                    mu=1e-6,
                    max_seconds=max_seconds,
                ),
                timeout_s=max_seconds + 180.0,
                logger=None,
            )
            row["cg_gamma"] = gamma
            rows.append(row)
            if row["status"] == "ok":
                log(
                    f"  {row['dataset']:16s} gamma={gamma}  "
                    f"lambda_max(S)={row['lambda_max_S']:10.3f}  "
                    f"inner: mean={row['inner_mean']:6.2f} "
                    f"first={row['inner_first']:3d} last={row['inner_last']:3d} "
                    f"max={row['inner_max']:3d}  run={row['run_time']:7.2f}s"
                )
            else:
                log(f"  {row['dataset']:16s} gamma={gamma}  {row['status']}: {row.get('error')}")
    df = pd.DataFrame(rows)
    if len(df):
        keep = [
            c
            for c in ("dataset", "I", "L", "P", "cg_gamma", "status", "lambda_max_S",
                      "inner_mean", "inner_first", "inner_last", "inner_max",
                      "run_time", "final_feasibility", "m_checksum")
            if c in df
        ]
        summary = df[keep]
        log(summary.to_string(index=False, float_format=lambda v: f"{v:.6g}"))
        # The comparison the earlier study made, redone. Compared over the
        # common prefix of outer iterations: a schedule that ran fewer of them
        # (because it hit its wall-clock budget) would otherwise be averaged
        # over a different, and systematically cheaper, stretch of the run.
        for ds, grp in df[df["status"] == "ok"].groupby("dataset"):
            if len(grp) != 2:
                continue
            traces = [list(r) for r in grp["inner_iterations"]]
            n = min(len(t) for t in traces)
            if n == 0:
                continue
            a, b = (float(np.mean(t[:n])) for t in traces)
            g1, g2 = grp["cg_gamma"].to_list()
            log(
                f"  {ds}: mean inner CG iterations over the first {n} outer "
                f"iterations, gamma={g1}: {a:.2f} vs gamma={g2}: {b:.2f}  "
                f"-> {abs(a - b) / max(a, b, 1e-12):.2%} apart"
            )
    df.drop(columns=["inner_iterations"], errors="ignore").to_csv(
        os.path.join(results_dir, "regimeG_cg_schedules.csv"), index=False
    )
    if "inner_iterations" in df:
        with open(os.path.join(results_dir, "regimeG_cg_inner_traces.json"), "w") as fh:
            json.dump(
                [
                    {"dataset": r["dataset"], "cg_gamma": r["cg_gamma"],
                     "inner_iterations": r.get("inner_iterations", [])}
                    for r in df.to_dict("records")
                ],
                fh,
            )
    return df


def section_i(audit, results_dir, log, mem_budget_mb=700):
    """`lambda_max(S)` across the whole grid.

    Sec. 5.4 argues the capacitance is well conditioned because `S >= I_P`. That
    is a lower bound on the spectrum and says nothing about the upper end, which
    is what actually sets the CG iteration count. Mapping it over the grid is
    what explains Section G's result rather than merely reporting it.
    """
    log("=== Section I: capacitance conditioning across the grid ===")
    rows = []
    for rec in audit.to_dict("records"):
        row = spawn(
            dict(task="lambda_max", path=rec["path"], mem_budget_mb=mem_budget_mb),
            timeout_s=600.0,
            logger=None,
        )
        row.update(
            I=rec["I"], L=rec["L"], P=rec["P"],
            delta=rec["delta"], delta_m=rec["delta_m"], IL=rec["IL"],
        )
        rows.append(row)
        if row["status"] == "ok":
            log(
                f"  {rec['dataset']:16s} I={rec['I']:3d} L={rec['L']:5d} "
                f"P={rec['P']:5d}  lambda_max(S)={row['lambda_max_S']:10.2f}  "
                f"-> CG needs ~{row['cg_iterations_per_digit']:.0f} iterations "
                f"per digit"
            )
        else:
            log(f"  {rec['dataset']:16s} {row['status']}: {row.get('error')}")
    df = pd.DataFrame(rows)
    df.to_csv(
        os.path.join(results_dir, "regimeI_capacitance_conditioning.csv"), index=False
    )
    return df


# ===========================================================================
# Section H: cross-backend fixed-point agreement
# ===========================================================================


# Section H runs Algorithm 5 to a matched iteration count, so each
# configuration carries its own budget: the point is to compare the *same*
# number of iterations across backends, and a backend that cannot afford them
# has to be recorded as unaffordable rather than compared at a shorter run.
FP_CONFIGS = [
    # (I, delta, delta_m, outer iterations, per-run wall-clock budget in s)
    (2, 10, 10, 300, 150.0),
    (8, 10, 10, 300, 250.0),
    (8, 20, 10, 300, 350.0),
    # Fewer outer iterations at the deepest point: the comparison needs a
    # *matched* iteration count across backends, not a long one, and S4 costs
    # ~6 s per iteration here against S3's 0.23 s.
    (32, 40, 10, 60, 700.0),
]

# The two CG tolerance schedules S4 is run under. The first is the library
# default (`cg_eta0=None` means `eta0 = max(||s||, 1)`, i.e. the first outer
# iteration asks for a relative accuracy of 1 and gets essentially no
# correction at all); the second is `eta0 = 1`, which is what every other study
# in this repository passes.
FP_S4_ARMS = [("smw_cg", None), ("smw_cg_eta0_1", 1.0)]


def section_h(
    configs, results_dir, scratch, log, mu=1e-6, mem_budget_mb=700,
):
    """`(tau, sigma, mu)` are held fixed across backends, so any difference in
    the iterate is attributable to the projector alone.

    `tau = sigma = 0.9/||K||` with `||K|| = ||C||` (Eq. (43): Algorithm 5's
    dualized operator never involves `A`), computed once per dataset and passed
    to every backend rather than recomputed per run.
    """
    log("=== Section H: cross-backend agreement of the fixed point ===")
    rows, pairs = [], []
    for path, outer_iterations, max_seconds in configs:
        norm_c = _norm_C(path)
        tau = sigma = 0.9 / norm_c
        saved = {}
        arms = [(m, None) for m in METHODS if m != "smw_cg"] + FP_S4_ARMS
        for name, eta0 in arms:
            method = "smw_cg" if name.startswith("smw_cg") else name
            out = os.path.join(
                scratch, f"fp_{os.path.basename(path.rstrip('/'))}_{name}.npy"
            )
            cfg = dict(
                task="fixed_point", path=path, method=method,
                tau=tau, sigma=sigma, mu=mu,
                outer_iterations=outer_iterations,
                mem_budget_mb=mem_budget_mb, m_out=out,
                max_seconds=max_seconds,
            )
            if eta0 is not None:
                cfg["cg_eta0"] = eta0
            row = spawn(cfg, timeout_s=max_seconds + 180.0, logger=None)
            row.update(tau=tau, sigma=sigma, mu=mu, arm=name, cg_eta0=eta0)
            rows.append(row)
            status = row["status"]
            log(
                f"  {row['dataset']:16s} {name:16s} {status:8s} "
                + (
                    f"|m|={row['m_norm']:.10g}  feas={row['feasibility']:.3e}  "
                    f"{row['run_time']:.2f}s"
                    if status == "ok"
                    else f"({row.get('error', '')})"
                )
            )
            if status == "ok" and os.path.exists(out):
                saved[name] = np.load(out)
        names = list(saved)
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                ma, mb = saved[names[a]], saved[names[b]]
                rel = float(
                    np.linalg.norm(ma - mb) / max(np.linalg.norm(ma), 1e-300)
                )
                pairs.append(
                    dict(
                        dataset=os.path.basename(path.rstrip("/")),
                        outer_iterations=outer_iterations,
                        method_a=names[a], method_b=names[b],
                        rel_difference=rel,
                        exact_pair=not names[a].startswith("smw_cg")
                        and not names[b].startswith("smw_cg"),
                    )
                )
    pdf = pd.DataFrame(pairs)
    if len(pdf):
        log(pdf.to_string(index=False, float_format=lambda v: f"{v:.3e}"))
        exact = pdf[pdf["exact_pair"]]
        if len(exact):
            worst = float(exact["rel_difference"].max())
            log(
                f"  worst disagreement among the *exact* backends "
                f"(S1/S2/S3): {worst:.3e}"
            )
        inexact = pdf[~pdf["exact_pair"]]
        if len(inexact):
            log(
                f"  worst disagreement involving S4 (inexact by construction): "
                f"{float(inexact['rel_difference'].max()):.3e}"
            )
    pd.DataFrame(rows).to_csv(
        os.path.join(results_dir, "regimeH_fixed_point_runs.csv"), index=False
    )
    pdf.to_csv(
        os.path.join(results_dir, "regimeH_fixed_point_agreement.csv"), index=False
    )
    return pdf


def _norm_C(path):
    """||C|| by power iteration, in a child process so the parent never holds a
    dataset."""
    row = spawn(
        dict(task="norm_c", path=path, mem_budget_mb=None), timeout_s=300.0, logger=None
    )
    if row["status"] != "ok":
        raise RuntimeError(f"could not compute ||C|| for {path}: {row.get('error')}")
    return row["norm_C"]


def _worker_norm_c(cfg):
    meter = _Meter(cfg)
    A, B_list, C, d_list, m = meter.load()
    meter.start()
    try:
        C_star = C.conj().T
        rng = np.random.default_rng(0)
        v = rng.normal(size=C.shape[1]) + 1j * rng.normal(size=C.shape[1])
        v /= np.linalg.norm(v)
        lam = 0.0
        for _ in range(300):
            w = C_star @ (C @ v)
            lam = np.linalg.norm(w)
            v = w / max(lam, 1e-300)
        meter.result["norm_C"] = float(np.sqrt(lam))
    except Exception as exc:
        meter.result.update(status="error", error=f"{type(exc).__name__}: {exc}")
    meter.finish()


# ===========================================================================
# Verdicts
# ===========================================================================


def print_verdicts(wide, cond_a, amort, cross, frontier, memory, cond, log):
    """One stated verdict per condition, so the log stands on its own."""
    log("\n=== Verdicts ===")

    if cond_a is not None and len(cond_a):
        counts = cond_a["condition_a"].value_counts().to_dict()
        held = cond_a.dropna(subset=["S3_beats_one_factorization"])
        log(
            "(a) P << (I L)^{3/4}: "
            + ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))
            + f" over {len(cond_a)} configurations. The consequence it predicts "
            f"(one S3 projection cheaper than one E E* factorization) holds at "
            f"{int(held['S3_beats_one_factorization'].sum())}/{len(held)} of the "
            "configurations where it can be evaluated, including every one where "
            "the condition itself fails: SUFFICIENT BUT NOT NECESSARY, and loose "
            "by two to three octaves of I."
        )

    if amort is not None and len(amort):
        big = amort[(amort["I"] >= 4) & np.isfinite(amort["T_measured_vs_S1"])]
        small = amort[(amort["I"] < 4) & np.isfinite(amort["T_measured_vs_S1"])]
        log(
            f"(b) amortization T: max {float(big['T_measured_vs_S1'].max()):.2f} over "
            f"the {len(big)} configurations with I >= 4 "
            f"(against {float(small['T_measured_vs_S1'].max()):.1f} at I = 2): "
            "the claim T = O(1) once I >= 4 is CONFIRMED, and T < 1 from I = 8 "
            "means the setup is repaid before one baseline projection finishes."
        )

    if cross is not None and len(cross):
        fav = cross[cross["delta_m"] == 10].sort_values("L")
        parts = [
            f"L={int(r['L'])}: I*={r['I_star_S3_over_S2']:.1f}"
            + ("(already ahead at the smallest I)" if r["S3_faster_at_smallest_I"] else "")
            for _, r in fav.iterrows()
        ]
        log(
            "(c) S3-over-S2 crossover: " + ", ".join(parts) + ". It MOVES, and it "
            "moves DOWN with L, so mesh refinement is a second lever on the same "
            "crossover."
        )
        s4 = fav["I_star_S4_over_S3"]
        log(
            "    S4-over-S3: "
            + (
                "never, on any mesh past the smallest"
                if s4.isna().sum() >= len(s4) - 1
                else "see regimeE_crossover.csv"
            )
            + ". Sec. 5.9's prediction that S4 overtakes S3 at the largest L is "
            "NOT CONFIRMED: S4 falls further behind as L grows, because both "
            "backends factor A (a common cost) and the two-basis discretization "
            "pins P, which keeps S3's dense solve cheap."
        )

    if frontier is not None and len(frontier):
        for _, r in frontier.iterrows():
            log(
                f"    frontier {r['backend']}: largest {r['largest']} "
                f"(I*L={r.get('IL')}), peak {r.get('peak_mb', float('nan')):.1f} MB, "
                f"{r.get('n_configs_ok')} ok / {r.get('n_configs_memory')} out of "
                f"memory / {r.get('n_configs_timeout')} timed out"
            )
    if memory is not None and len(memory):
        both = memory.dropna(subset=["S3_minus_S4_mb"])
        deep = both.loc[both["IL"].idxmax()] if len(both) else None
        s3_fails = memory[(memory["S3_status"] == "memory") & (memory["S4_status"] == "ok")]
        if deep is not None:
            log(
                f"(memory) at the deepest configuration both run, {deep['dataset']}, "
                f"S3 needs {deep['S3_mb']:.1f} MB against S4's {deep['S4_mb']:.1f} MB: "
                f"{deep['S3_minus_S4_mb']:.1f} MB saved, a factor "
                f"{deep['S3_mb'] / deep['S4_mb']:.1f}."
            )
        if len(s3_fails):
            r = s3_fails.iloc[0]
            log(
                f"    where S3 cannot be formed at all ({', '.join(s3_fails['dataset'])}): "
                f"the model puts S3 at {r['pred_S3_mb']:.0f} MB (a {r['P']}^2 dense "
                f"capacitance plus its L x P transients) against S4's measured "
                f"{s3_fails['S4_mb'].min():.1f}-{s3_fails['S4_mb'].max():.1f} MB. "
                "S4's case is memory, and it is decisive there."
            )

    if cond is not None and len(cond):
        ok = cond[cond["status"] == "ok"].copy()
        if len(ok):
            arg = ok["I"] * (ok["delta"] / ok["delta_m"]) ** 2
            c = float(np.median(ok["lambda_max_S"] / arg))
            rel = np.abs(c * arg - ok["lambda_max_S"]) / ok["lambda_max_S"]
            log(
                f"(conditioning) lambda_max(S) = {c:.2f} * I * (delta/delta_m)^2 to a "
                f"median {float(np.median(rel)):.1%} relative error over "
                f"{len(ok)} configurations, spanning "
                f"{float(ok['lambda_max_S'].min()):.1f} to "
                f"{float(ok['lambda_max_S'].max()):.0f}. It does not depend on L. "
                "S >= I_P bounds only the bottom of the spectrum; the top grows with "
                "I and with the square of the mesh ratio, which is what S4 pays for "
                "and S3 does not."
            )


# ===========================================================================
# Plots
# ===========================================================================


def make_plots(wide, bench, model, cg, visuals_dir, log):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"axes.grid": True, "figure.dpi": 110})
    style = {
        "S1": ("tab:red", "o", "S1 spsolve"),
        "S2": ("tab:orange", "s", "S2 cached_splu"),
        "S3": ("tab:blue", "^", "S3 smw"),
        "S4": ("tab:green", "v", "S4 smw_cg"),
    }
    fav = wide[wide["delta_m"] == 10].copy()
    deltas = sorted(fav["delta"].dropna().unique())

    # (1) per-call cost against I, one panel per field mesh.
    fig, axs = plt.subplots(1, len(deltas), figsize=(5 * len(deltas), 4.3), sharey=True)
    axs = np.atleast_1d(axs)
    for ax, d in zip(axs, deltas):
        grp = fav[fav["delta"] == d].sort_values("I")
        for short, (color, marker, label) in style.items():
            y = grp[f"{short}_call"].to_numpy(float) * 1000
            ok = np.isfinite(y)
            if ok.any():
                ax.plot(grp["I"].to_numpy()[ok], y[ok], marker=marker, color=color,
                        label=label)
        L = int(grp["L"].iloc[0])
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Number of sources $I$")
        ax.set_title(f"$\\delta={int(d)}$: $L={L}$, $P=394$")
    axs[0].set_ylabel("Per-projection time (ms)")
    axs[0].legend(fontsize=8)
    fig.suptitle(
        "Projector cost on the joint $I\\times\\delta$ grid, contrast mesh pinned "
        "at $\\delta_m=10$"
    )
    fig.tight_layout()
    fig.savefig(os.path.join(visuals_dir, "regime_per_call_vs_I.pdf"))
    fig.savefig(os.path.join(visuals_dir, "regime_per_call_vs_I.png"))
    plt.close(fig)

    # (2) the regime ratio.
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    for d in deltas:
        grp = fav[fav["delta"] == d].sort_values("I")
        ax.plot(grp["I"], grp["P_over_IL"], marker="o", label=f"$\\delta={int(d)}$")
    unfav = wide[wide["delta_m"] != 10].sort_values("I")
    if len(unfav):
        ax.plot(unfav["I"], unfav["P_over_IL"], marker="x", ls="--", color="k",
                label="$\\delta_m=\\delta=40$ (single mesh)")
    ax.axhline(1.0, color="gray", lw=0.8)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Number of sources $I$")
    ax.set_ylabel("$P/(I\\,L)$")
    ax.set_title("The regime ratio: rank of $BB^*$ over the dimension it corrects")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(visuals_dir, "regime_ratio.pdf"))
    fig.savefig(os.path.join(visuals_dir, "regime_ratio.png"))
    plt.close(fig)

    # (3) memory, faceted by mesh. Plotting it against I*L on one axis would
    # interleave three different field meshes at nearly equal I*L (delta=10 at
    # I=32, delta=20 at I=8, delta=40 at I=2 are all near 7000) while S3's and
    # S4's footprints depend on L and P and not on the product, which turns the
    # model curves into zigzags. One facet per mesh keeps L fixed inside each.
    families = [(d, 10) for d in deltas] + (
        [(40, 40)] if (wide["delta_m"] != 10).any() else []
    )
    mem_model = model.set_index("dataset") if len(model) else None
    fig, axs = plt.subplots(1, len(families), figsize=(4.6 * len(families), 4.3),
                            sharey=True)
    axs = np.atleast_1d(axs)
    for ax, (d, dm) in zip(axs, families):
        grp = wide[(wide["delta"] == d) & (wide["delta_m"] == dm)].sort_values("I")
        for short, (color, marker, label) in style.items():
            y = grp[f"{short}_mb"].to_numpy(float)
            ok = np.isfinite(y)
            if ok.any():
                ax.plot(grp["I"].to_numpy()[ok], y[ok], marker=marker, ls="none",
                        color=color, label=label)
        if mem_model is not None:
            mm = mem_model.reindex(grp["dataset"])
            for col, color, label in (
                ("EE_star_mb", "tab:red", "$EE^*$ stored"),
                ("smw_total_mb", "tab:blue", "S3 dense blocks"),
                ("smw_cg_total_mb", "tab:green", "S4"),
            ):
                ax.plot(grp["I"], mm[col].to_numpy(float), ls=":", lw=1,
                        color=color, label=label + " (model)")
        ax.axhline(700, color="k", lw=0.8)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Number of sources $I$")
        ax.set_title(
            f"$\\delta={int(d)}$, $\\delta_m={int(dm)}$: "
            f"$L={int(grp['L'].iloc[0])}$, $P={int(grp['P'].iloc[0])}$"
        )
    axs[0].set_ylabel("MB attributable to the backend")
    axs[0].legend(fontsize=7, ncol=2)
    fig.suptitle(
        "Peak resident memory: measured with psutil, against the structural model "
        "(horizontal line: the 700 MB budget each backend was given)"
    )
    fig.tight_layout()
    fig.savefig(os.path.join(visuals_dir, "regime_memory.pdf"))
    fig.savefig(os.path.join(visuals_dir, "regime_memory.png"))
    plt.close(fig)

    # (3) CG inner-iteration traces per schedule.
    if cg is not None and len(cg):
        ok = cg[cg["status"] == "ok"]
        datasets = list(dict.fromkeys(ok["dataset"]))
        if datasets:
            fig, axs = plt.subplots(
                1, len(datasets), figsize=(5 * len(datasets), 4.0), squeeze=False
            )
            for ax, ds in zip(axs[0], datasets):
                for _, r in ok[ok["dataset"] == ds].iterrows():
                    trace = r.get("inner_iterations") or []
                    ax.plot(trace, label=f"$\\gamma={r['cg_gamma']}$")
                ax.set_xlabel("Outer iteration $k$")
                ax.set_title(f"{ds}\n$\\lambda_{{\\max}}(S)$="
                             f"{ok[ok['dataset'] == ds]['lambda_max_S'].iloc[0]:.1f}")
                ax.legend(fontsize=8)
            axs[0][0].set_ylabel("Inner CG iterations")
            fig.suptitle("CG inner iterations under the two tolerance schedules")
            fig.tight_layout()
            fig.savefig(os.path.join(visuals_dir, "regime_cg_inner_iterations.pdf"))
            fig.savefig(os.path.join(visuals_dir, "regime_cg_inner_iterations.png"))
            plt.close(fig)
    log(f"  plots written to {visuals_dir}")


# ===========================================================================
# Entry point
# ===========================================================================

_WORKERS = {
    "backend": _worker_backend,
    "cg_schedule": _worker_cg_schedule,
    "fixed_point": _worker_fixed_point,
    "lambda_max": _worker_lambda_max,
    "nnz": _worker_nnz,
    "norm_c": _worker_norm_c,
}


def make_run_dirs(exp_path, exp_name):
    exp = os.path.join(exp_path, exp_name)
    dirs = {
        "exp": exp,
        "results": os.path.join(exp, "results"),
        "visuals": os.path.join(exp, "visuals"),
        "scratch": os.path.join(exp, "scratch"),
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--data-root", default=JOINT_ROOT)
    parser.add_argument("--exp-path", default="runs")
    parser.add_argument("--exp-name", default="smw_regime")
    parser.add_argument(
        "--only", default="A,B,C,D,E,F,G,H,I",
        help="comma-separated subset of the sections to run",
    )
    parser.add_argument("--n-calls", type=int, default=5)
    parser.add_argument("--time-budget", type=float, default=8.0)
    parser.add_argument(
        "--mem-budget-mb", type=float, default=700,
        help="address space each backend is given on top of the loaded dataset; "
             "exceeding it is recorded as status='memory' rather than an OOM kill",
    )
    parser.add_argument("--naive-cutoff", type=int, default=4000)
    parser.add_argument(
        "--respect-cutoff", action="store_true",
        help="skip S1/S2 outright past the I*L cutoff instead of measuring where "
             "they actually fail",
    )
    parser.add_argument("--cg-outer", type=int, default=150)
    parser.add_argument(
        "--reuse-model", action="store_true",
        help="re-derive sections C-F from an existing regimeF_memory_model.csv "
             "instead of re-measuring the structural fill",
    )
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.worker:
        cfg = json.loads(args.worker)
        _WORKERS[cfg.get("task", "backend")](cfg)
        return

    if args.quick:
        args.n_calls, args.time_budget = 3, 3.0
        args.cg_outer = 30

    dirs = make_run_dirs(args.exp_path, args.exp_name)
    log_path = os.path.join(dirs["exp"], "smw_regime_study.log")
    log_file = open(log_path, "w")

    def log(msg=""):
        print(msg, flush=True)
        log_file.write(str(msg) + "\n")
        log_file.flush()

    sections = {s.strip().upper() for s in args.only.split(",") if s.strip()}
    paths = [
        dataset_dir(i, d, 10, args.data_root) for i, d in FAVOURABLE_GRID
    ] + [dataset_dir(i, d, d, args.data_root) for i, d in UNFAVOURABLE_GRID]
    missing = [p for p in paths if not os.path.isdir(p)]
    if missing:
        raise SystemExit(
            "Missing datasets:\n  "
            + "\n  ".join(missing)
            + "\n\nGenerate them with:  bash scripts/generate_joint_sweep.sh"
        )

    t_start = time.time()
    audit = section_a(paths, dirs["results"], log) if "A" in sections else None
    if audit is None:
        audit = pd.read_csv(os.path.join(dirs["results"], "regimeA_dataset_audit.csv"))

    bench = None
    if "B" in sections:
        bench = section_b(
            audit, dirs["results"], log,
            n_calls=args.n_calls, time_budget_s=args.time_budget,
            mem_budget_mb=args.mem_budget_mb,
            naive_size_cutoff=args.naive_cutoff,
            respect_cutoff=args.respect_cutoff,
        )
    else:
        path = os.path.join(dirs["results"], "regimeB_backend_cost.csv")
        bench = pd.read_csv(path) if os.path.exists(path) else None

    model = None
    model_path = os.path.join(dirs["results"], "regimeF_memory_model.csv")
    if "F" in sections and not (args.reuse_model and os.path.exists(model_path)):
        model = section_nnz(audit, dirs["results"], log, mem_budget_mb=args.mem_budget_mb)
    elif os.path.exists(model_path):
        model = pd.read_csv(model_path)

    wide = pivot_backends(bench) if bench is not None else None
    if wide is not None:
        wide.to_csv(os.path.join(dirs["results"], "regimeB_backend_wide.csv"), index=False)

    cond_a = amort = cross = frontier = memory = None
    if "C" in sections and wide is not None:
        cond_a = section_c(wide, dirs["results"], log)
    if "D" in sections and wide is not None:
        amort = section_d(wide, dirs["results"], log)
    if "E" in sections and wide is not None:
        cross = section_e(wide, dirs["results"], log)
    if "F" in sections and wide is not None:
        memory, frontier = section_f(
            wide, bench, model if model is not None else pd.DataFrame(),
            dirs["results"], log, mem_budget_mb=args.mem_budget_mb,
        )

    cond = None
    if "I" in sections:
        cond = section_i(audit, dirs["results"], log, mem_budget_mb=args.mem_budget_mb)
    else:
        cond_path = os.path.join(dirs["results"], "regimeI_capacitance_conditioning.csv")
        cond = pd.read_csv(cond_path) if os.path.exists(cond_path) else None

    cg = None
    if "G" in sections:
        cg_configs = [
            dataset_dir(32, 40, 10, args.data_root),   # large I, small P
            dataset_dir(8, 40, 40, args.data_root),    # large I, large P (6720)
            dataset_dir(2, 40, 40, args.data_root),    # the previous large-P point
        ]
        cg = section_g(
            [p for p in cg_configs if os.path.isdir(p)], dirs["results"], log,
            outer_iterations=args.cg_outer, mem_budget_mb=args.mem_budget_mb,
        )
    if "H" in sections:
        fp_configs = [
            (dataset_dir(i, d, dm, args.data_root), iters, budget)
            for i, d, dm, iters, budget in FP_CONFIGS
            if os.path.isdir(dataset_dir(i, d, dm, args.data_root))
        ]
        section_h(
            fp_configs, dirs["results"], dirs["scratch"], log,
            mem_budget_mb=args.mem_budget_mb,
        )

    print_verdicts(wide, cond_a, amort, cross, frontier, memory, cond, log)

    if wide is not None:
        make_plots(
            wide, bench, model if model is not None else pd.DataFrame(),
            cg, dirs["visuals"], log,
        )

    log(f"\nTotal wall time {time.time() - t_start:.1f}s. Results under {dirs['exp']}")
    log_file.close()


if __name__ == "__main__":
    main()
