import os

import numpy as np
import scipy.sparse as sp
import torch

from iwp.algorithms.algorithms import (
    FISTA,
    ClosedFormSolution,
    NesterovAcceleratedGradientDescent,
    DistributedBlockGradientDescent
)
from iwp.algorithms.plot import plot_all_algorithms_convergence
from iwp.data.export import export_all_metrics_to_csv, save_complex_vector
from iwp.data.load_experiment_data import load_experiment_data
from iwp.utils.config import load_yaml_into_namespace, parse_arguments
from iwp.utils.logger import setup_logger
from iwp.utils.utils import copy_file, make_dirs, set_seed

if __name__ == "__main__":

    args = parse_arguments()
    args = load_yaml_into_namespace(args.config, args)

    # Paths to save the results
    args.exp_path, args.visuals_path, args.results_path = make_dirs(
        os.path.join(args.exp_path, args.exp_name)
    )

    logger = setup_logger(
        name="iwp",
        log_file=(
            os.path.join(args.exp_path, f"{args.exp_name}.log")
            if args.save_logs
            else None
        ),
        level=args.log_level,
        log_to_console=args.log_to_console,
    )

    # Save the config file in the experiment folder
    copy_file(args.config, os.path.join(args.exp_path, "config.yaml"))

    # Setup device and seed
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA is not available. Falling back to CPU.")
        args.device = "cpu"
    if args.device not in ["cpu", "cuda"]:
        raise ValueError("Device must be either 'cpu' or 'cuda'.")
    logger.info(f"Using device: {args.device}")
    set_seed(args.seed)

    # Load data
    A, B_list, C, d_list, m = load_experiment_data(args.data_path)
    A_star = A.conj().T
    C_star = C.conj().T

    I = len(B_list)
    J, L = C.shape
    P = B_list[0].shape[1]
    logger.info(f"Dimensions: I={I}, J={J}, L={L}, P={P}")

    row_blocks = []
    for i in range(I):
        blocks = [sp.csr_matrix((J, L))] * I + [sp.csr_matrix((J, P))]
        blocks[i] = sp.csr_matrix(C)
        row_blocks.append(sp.hstack(blocks, format="csr"))
    D = sp.vstack(
        row_blocks, format="csr"
    )  # shape: (I*J, I*L + P) if A is (L,L), B_i is (L,P)
    D_star = D.conj().T

    d = np.concatenate(d_list, axis=0)  # shape: (I*J,)

    row_blocks = []
    for i in range(I):
        blocks = [sp.csr_matrix((L, L))] * I + [-B_list[i]]
        blocks[i] = sp.csr_matrix(A)
        row_blocks.append(sp.hstack(blocks, format="csr"))
    E = sp.vstack(row_blocks, format="csr")  # shape: (I*L, I*L + P)
    E_star = E.conj().T

    lambd = float(args.lambd)
    mu_1 = float(args.mu_1)
    mu_2 = float(args.mu_2)
    mu_3 = float(args.mu_3)

    K_eigenvalues_A = np.linalg.eigvals((A.conj().T @ A).toarray())
    K_A = np.max(np.abs(K_eigenvalues_A))
    logger.info(f"Lipschitz constant K_A: {K_A}")

    def get_J_1(d, lambd, mu):
        def J_1(x):
            Dx_minus_d = D @ x - d
            Ex = E @ x
            return (
                0.5 * np.vdot(Dx_minus_d, Dx_minus_d).real
                + 0.5 * lambd * np.vdot(Ex, Ex).real
                + 0.5 * mu * np.vdot(x, x).real
            )

        return J_1

    def get_dJ_1(d, lambd, mu):
        def dJ_1(x):
            Dx_minus_d = D @ x - d
            Ex = E @ x
            return D.conj().T @ Dx_minus_d + lambd * E_star @ Ex + mu * x

        return dJ_1

    def get_closed_form_solution_J_1(d, lambd, mu):
        def closed_form_solution_J_1():
            return sp.linalg.spsolve(
                D.conj().T @ D + lambd * E_star @ E + mu * sp.eye(I * L + P),
                D.conj().T @ d,
            )

        return closed_form_solution_J_1

    def get_K_J_1(lambd, mu):
        K_op_J_1 = D.conj().T @ D + lambd * E_star @ E + mu * sp.eye(I * L + P)
        K_eigenvalues_J_1 = np.linalg.eigvals(K_op_J_1.toarray())
        return np.max(np.abs(K_eigenvalues_J_1))

    def get_J_2(d, mu, threshold=1e-6):
        def J_2(x):
            Dx_minus_d = D @ x - d
            Ex = E @ x
            if np.linalg.norm(Ex) < threshold:
                return (
                    0.5 * np.vdot(Dx_minus_d, Dx_minus_d).real
                    + 0.5 * mu * np.vdot(x[-P:], x[-P:]).real
                )
            else:
                return np.inf

        return J_2

    def get_grad_J_2(d, mu):
        def grad_J_2(x):
            reg = np.zeros_like(x)
            reg[-P:] = mu * x[-P:]
            return D_star @ (D @ x - d) + reg

        return grad_J_2

    def get_prox_J_2():
        def prox_J_2(x, gamma):
            w = sp.linalg.spsolve(E @ E_star, E @ x)
            return x - E_star @ w

        return prox_J_2

    def get_K_J_2(mu):
        K_op_J_2 = D.conj().T @ D + mu * sp.eye(I * L + P)
        K_eigenvalues_J_2 = np.linalg.eigvals(K_op_J_2.toarray())
        return np.max(np.abs(K_eigenvalues_J_2))

    def get_J_3(d_list, mu):
        def J_3(m):
            total = 0.0
            for i in range(len(B_list)):
                CA_inv_Bi_m = C @ sp.linalg.spsolve(A, B_list[i] @ m)
                diff = CA_inv_Bi_m - d_list[i]
                total += 0.5 * np.vdot(diff, diff).real
            return total + 0.5 * mu * np.vdot(m, m).real

        return J_3

    def get_dJ_3(d_list, mu):
        def dJ_3(m):
            p_sum = sum(
                B_i.conj().T
                @ sp.linalg.spsolve(
                    A_star,
                    C_star @ (C @ sp.linalg.spsolve(A, B_i @ m) - d_i),
                )
                for B_i, d_i in zip(B_list, d_list)
            )
            return p_sum + mu * m

        return dJ_3

    def get_K_J_3(mu):
        Ainv = sp.linalg.inv(A.tocsc())
        K_op_J_3 = sum(
            B_i.conj().T @ Ainv.conj().T @ C.conj().T @ C @ Ainv @ B_i for B_i in B_list
        ) + mu * sp.eye(P)
        K_eigenvalues_J_3 = np.linalg.eigvals(K_op_J_3.toarray())
        return np.max(np.abs(K_eigenvalues_J_3))

    # Distributed Gradient Descent
    def get_grad_u(d_list, rho):
        def grad_u(i,u_i,m):
            return C_star @ (C @ u_i - d_list[i]) + rho * (A_star @ (A @ u_i - B_list[i] @ m))
        return grad_u

    def get_grad_m(agent_indices, dg, rho):
        def grad_m(s, m, u_list):
            grad_sum = np.zeros_like(m)
            for i in agent_indices[s]:
                grad_sum += B_list[i].conj().T @ (B_list[i] @ m - A @ u_list[i])
            return rho * grad_sum + (1.0 / S) * dg(m)
        return grad_m

    # Tikhonov gradient:
    def dg_tikhonov(mu=1.0):
        return lambda m: mu * m
    # ================================ No noise case ================================
    '''
    # Get functions and Lipschitz constants
    J_1 = get_J_1(d, lambd, mu_1)
    dJ_1 = get_dJ_1(d, lambd, mu_1)
    closed_form_solution_J_1 = get_closed_form_solution_J_1(d, lambd, mu_1)
    K_J_1 = get_K_J_1(lambd, mu_1)
    logger.info(f"Lipschitz constant K_J_1: {K_J_1}")

    J_2 = get_J_2(d, mu_2)
    grad_J_2 = get_grad_J_2(d, mu_2)
    prox_J_2 = get_prox_J_2()
    K_J_2 = get_K_J_2(mu_2)
    logger.info(f"Lipschitz constant K_J_2: {K_J_2}")

    J_3 = get_J_3(d_list, mu_3)
    dJ_3 = get_dJ_3(d_list, mu_3)
    K_J_3 = get_K_J_3(mu_3)
    logger.info(f"Lipschitz constant K_J_3: {K_J_3}")
    '''
    x_0 = np.zeros(I * L + P, dtype=np.complex128)  # shape: (I*L + P,)
    max_iterations = int(args.max_iterations)
    '''
    # Run algorithms
    algo_1 = ClosedFormSolution(
        exp_name=args.exp_name,
        algo_plot_name="P-ClosedForm",
        f=J_1,
        solution=closed_form_solution_J_1,
        logger=logger,
        verbose=args.verbose,
    )
    x_1 = algo_1.run(x0=x_0, max_iterations=max_iterations)
    algo_1.plot_algorithm_convergence(m, args.visuals_path)
    export_all_metrics_to_csv(
        algo_1, os.path.join(args.results_path, "P-ClosedForm_Metrics.csv")
    )
    save_complex_vector(
        os.path.join(args.results_path, "P-ClosedForm_PredictedVectorm.dat"), x_1[-P:]
    )

    algo_2 = FISTA(
        exp_name=args.exp_name,
        algo_plot_name="FISTA",
        f=J_2,
        grad=grad_J_2,
        prox=prox_J_2,
        K=K_J_2,
        logger=logger,
        verbose=args.verbose,
    )
    x_2 = algo_2.run(x0=x_0, max_iterations=max_iterations)
    algo_2.plot_algorithm_convergence(m, args.visuals_path)
    export_all_metrics_to_csv(
        algo_2, os.path.join(args.results_path, "FISTA_Metrics.csv")
    )
    save_complex_vector(
        os.path.join(args.results_path, "FISTA_PredictedVectorm.dat"), x_2[-P:]
    )

    algo_3 = NesterovAcceleratedGradientDescent(
        exp_name=args.exp_name,
        algo_plot_name="C-NAGD",
        f=J_3,
        df=dJ_3,
        K=K_J_3,
        logger=logger,
        verbose=args.verbose,
    )
    m_3 = algo_3.run(x0=x_0[-P:], max_iterations=max_iterations)
    algo_3.plot_algorithm_convergence(m, args.visuals_path)
    export_all_metrics_to_csv(
        algo_3, os.path.join(args.results_path, "C-NAGD_Metrics.csv")
    )
    save_complex_vector(
        os.path.join(args.results_path, "C-NAGD_PredictedVectorm.dat"), m_3
    )
    '''

    # create agent_indices (round-robin)
    S = 2  # choose number of agents
    agent_indices = [[i for i in range(s, I, S)] for s in range(S)]

    # mixing matrix: simple symmetric averaging
    W = np.full((S, S), 1.0 / S)

    rho = 1.0
    alpha = 1e-3  # step size — tune (must be small enough; use Lipschitz if available) Will do estimation later
    grad_u = get_grad_u(d_list, rho)
    dg = dg_tikhonov(mu=mu_1)  # or your TV / dg
    grad_m = get_grad_m(agent_indices, dg, rho)
    J_1 = get_J_1(d, lambd, mu_1)

    algo_dbgd = DistributedBlockGradientDescent(
        exp_name=args.exp_name,
        algo_plot_name="D-BGD",
        f=lambda x: J_1(x),
        grad_u=grad_u,
        grad_m=grad_m,
        S=S,
        W=W,
        alpha=alpha,
        P=P,
        L=L,
        I=I,
        agent_indices=agent_indices,
        use_mpi=args.use_mpi if hasattr(args, "use_mpi") else False,  # set via config or CLI; requires mpi4py and mpirun
        logger=logger,
        verbose=args.verbose,
    )
    
    # Log agent index distribution and per-rank ownership for diagnostics
    try:
        total_indices = sum(len(lst) for lst in agent_indices)
        counts = [len(lst) for lst in agent_indices]
    except Exception:
        total_indices = None
        counts = None

    if algo_dbgd.use_mpi:
        # rank 0 prints the global distribution
        if algo_dbgd.rank == 0:
            logger.info(f"Agent indices distribution (total={total_indices}): {counts}")
        # every rank prints which local agents it owns and how many indices that corresponds to
        local_agents = algo_dbgd.local_agents if algo_dbgd.local_agents is not None else []
        local_count = sum(len(agent_indices[s]) for s in local_agents) if counts is not None else None
        logger.info(f"Rank {algo_dbgd.rank}/{algo_dbgd.world_size} local_agents={local_agents} -> {local_count} indices")
    else:
        logger.info(f"Running without MPI. Agent indices distribution (total={total_indices}): {counts}")

    x_final = algo_dbgd.run(x0=x_0, max_iterations=max_iterations)
    m_dbgd = x_final[-P:]

    # Only let rank 0 (or the single non-MPI run) produce plots and write results
    if not algo_dbgd.use_mpi or algo_dbgd.rank == 0:
        algo_dbgd.plot_algorithm_convergence(m, args.visuals_path)
        export_all_metrics_to_csv(
            algo_dbgd, os.path.join(args.results_path, "D-BGD_Metrics.csv")
        )
        save_complex_vector(
            os.path.join(args.results_path, "D-BGD_PredictedVectorm.dat"), m_dbgd
        )
        plot_all_algorithms_convergence(
            algorithms=[algo_dbgd],
            visuals_path=args.visuals_path,
            show=False,
            save=True,
            show_time_memory=True,
            results=True,
        )
'''
    plot_all_algorithms_convergence(
        algorithms=[algo_1, algo_2, algo_3, algo_dbgd],
        visuals_path=args.visuals_path,
        show=False,
        save=True,
        show_time_memory=True,
        results=True,
    )
'''

'''
    # ================================ Noise case ================================

    if args.enable_noise:
        noise_level = float(args.noise_level)
        samples = int(args.samples)
        logger.info(
            f"Adding Gaussian noise with standard deviation {noise_level} for {samples} samples."
        )

        # Initialize arrays for metrics: shape (3, samples)
        mse_array = np.zeros((3, samples))
        mae_array = np.zeros((3, samples))
        f_array = np.zeros((3, samples))

        for idx, sample in enumerate(range(samples)):
            logger.info(f"Sample {sample + 1}/{samples}")

            # Add noise to the observations
            d_list_noisy = [
                d_i
                + noise_level
                * (
                    np.random.normal(size=d_i.shape)
                    + 1j * np.random.normal(size=d_i.shape)
                )
                for d_i in d_list
            ]
            d_noisy = np.concatenate(d_list_noisy, axis=0)

            # Get functions and keep the same Lipschitz constants
            J_1_noisy = get_J_1(d_noisy, lambd, mu_1)
            dJ_1_noisy = get_dJ_1(d_noisy, lambd, mu_1)
            closed_form_solution_J_1_noisy = get_closed_form_solution_J_1(
                d_noisy, lambd, mu_1
            )

            J_2_noisy = get_J_2(d_noisy, mu_2)
            grad_J_2_noisy = get_grad_J_2(d_noisy, mu_2)
            prox_J_2_noisy = get_prox_J_2()

            J_3_noisy = get_J_3(d_list_noisy, mu_2)
            dJ_3_noisy = get_dJ_3(d_list_noisy, mu_2)

            # Run algorithms
            algo_1_noisy = ClosedFormSolution(
                exp_name=args.exp_name,
                algo_plot_name="P-ClosedForm",
                f=J_1_noisy,
                solution=closed_form_solution_J_1_noisy,
                logger=logger,
                verbose=args.verbose,
            )

            x_1_noisy = algo_1_noisy.run(x0=x_0, max_iterations=max_iterations)
            algo_1_noisy.plot_algorithm_convergence(
                m, args.visuals_path, show=False, save=False
            )
            if idx == 0:
                save_complex_vector(
                    os.path.join(
                        args.results_path,
                        f"P-ClosedForm_noise={noise_level}_PredictedVectorm.dat",
                    ),
                    x_1_noisy[-P:],
                )
            mse_array[0, sample] = algo_1_noisy.mse_values[-1]
            mae_array[0, sample] = algo_1_noisy.mae_values[-1]
            f_array[0, sample] = algo_1_noisy.f_values[-1]

            algo_2_noisy = FISTA(
                exp_name=args.exp_name,
                algo_plot_name="FISTA",
                f=J_2_noisy,
                grad=grad_J_2_noisy,
                prox=prox_J_2_noisy,
                K=K_J_2,
                logger=logger,
                verbose=args.verbose,
            )
            x_2_noisy = algo_2_noisy.run(x0=x_0, max_iterations=max_iterations)
            algo_2_noisy.plot_algorithm_convergence(
                m, args.visuals_path, show=False, save=False
            )
            if idx == 0:
                save_complex_vector(
                    os.path.join(
                        args.results_path,
                        f"FISTA_noise={noise_level}_PredictedVectorm.dat",
                    ),
                    x_2_noisy[-P:],
                )
            mse_array[1, sample] = algo_2_noisy.mse_values[-1]
            mae_array[1, sample] = algo_2_noisy.mae_values[-1]
            f_array[1, sample] = algo_2_noisy.f_values[-1]

            algo_3_noisy = NesterovAcceleratedGradientDescent(
                exp_name=args.exp_name,
                algo_plot_name="C-NAGD",
                f=J_3_noisy,
                df=dJ_3_noisy,
                K=K_J_3,
                logger=logger,
                verbose=args.verbose,
            )
            m_3_noisy = algo_3_noisy.run(x0=x_0[-P:], max_iterations=max_iterations)
            algo_3_noisy.plot_algorithm_convergence(
                m, args.visuals_path, show=False, save=False
            )
            if idx == 0:
                save_complex_vector(
                    os.path.join(
                        args.results_path,
                        f"C-NAGD_noise={noise_level}_PredictedVectorm.dat",
                    ),
                    m_3_noisy,
                )
            mse_array[2, sample] = algo_3_noisy.mse_values[-1]
            mae_array[2, sample] = algo_3_noisy.mae_values[-1]
            f_array[2, sample] = algo_3_noisy.f_values[-1]

        # Compute mean and std for each metric and algorithm
        algo_names = ["P-ClosedForm", "FISTA", "C-NAGD"]
        metrics = ["MSE", "MAE", "f"]
        arrays = [mse_array, mae_array, f_array]

        for arr, metric in zip(arrays, metrics):
            mean = arr.mean(axis=-1)
            std = arr.std(axis=-1)
            for i, algo in enumerate(algo_names):
                logger.info(f"{algo} {metric}: mean={mean[i]:.6f}, std={std[i]:.6f}")
'''