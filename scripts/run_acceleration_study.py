#!/usr/bin/env python
"""Run only the acceleration / line-search study of Sec. 4.8.

`scripts/compare_algorithms.py` runs the full protocol, which takes a long
time and re-derives results that have not changed. This driver runs the three
new sections on their own:

  10  Tikhonov arm, every variant against C-NAGD and FISTA at equal work
  10b TV arm at the weight where the unaccelerated scheme was slowest
  10c fixed-point agreement, the acceptance check that acceleration changes
      the rate and nothing else

Run:
    pixi run python scripts/run_acceleration_study.py [--quick] [--mu 1e-3]
"""

import argparse
import os
import sys

# OpenBLAS threading has to be pinned *before* numpy is imported, so this sits
# above every other import on purpose. On the WSL2 box these numbers were
# measured on, the multi-threaded path is catastrophically slow for the small
# dense problems here: a 394x394 Hermitian eigendecomposition takes 0.19 s on
# one thread and 13 s on four, and a dense matvec is 40x slower. Since the
# dense blocks in this study are all a few hundred wide, threading has nothing
# to win and everything to lose. Override with --blas-threads if your machine
# behaves differently.
_BLAS_VARS = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
if "--blas-threads" in sys.argv:
    _threads = sys.argv[sys.argv.index("--blas-threads") + 1]
else:
    _threads = "1"
for _var in _BLAS_VARS:
    os.environ.setdefault(_var, _threads)

import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from compare_algorithms import (  # noqa: E402
    make_run_dirs,
    run_acceleration_fixed_point_check,
    run_acceleration_linesearch,
    run_acceleration_tv,
)
from iwp.experiments.comparison import load_problem  # noqa: E402
from iwp.utils.logger import setup_logger  # noqa: E402
from iwp.utils.utils import set_seed  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="data")
    parser.add_argument("--exp-path", default="runs")
    parser.add_argument("--exp-name", default="acceleration_study")
    parser.add_argument("--mu", type=float, default=1e-3)
    parser.add_argument("--lambda-tv", type=float, default=1e-3)
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--fp-iterations", type=int, default=60000)
    parser.add_argument("--max-seconds", type=float, default=None,
                        help="per-run wall-clock cap, useful for a smoke run")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-tv", action="store_true")
    parser.add_argument("--skip-fixed-point", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--blas-threads", default="1",
                        help="read before numpy is imported; see the note at the "
                             "top of this file")
    args = parser.parse_args()

    if args.quick:
        args.iterations, args.fp_iterations = 800, 2000

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

    summary = {}
    _, summary["tikhonov"], _, _ = run_acceleration_linesearch(
        pb, dirs, logger, mu=args.mu, max_iterations=args.iterations,
        max_seconds=args.max_seconds,
    )
    if not args.skip_tv:
        _, summary["tv"], _ = run_acceleration_tv(
            pb, dirs, logger, lambda_tv=args.lambda_tv, mu_extra=args.mu,
            max_iterations=args.iterations, max_seconds=args.max_seconds,
        )
    if not args.skip_fixed_point:
        summary["fixed_point"] = run_acceleration_fixed_point_check(
            pb, dirs, logger, mu=args.mu, max_iterations=args.fp_iterations
        )

    for name, df in summary.items():
        if isinstance(df, pd.DataFrame):
            logger.info(f"--- {name} ---\n{df.to_string(index=False)}")
    logger.info(f"Results under {dirs['exp']}")


if __name__ == "__main__":
    main()
