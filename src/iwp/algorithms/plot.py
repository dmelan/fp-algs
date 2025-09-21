import math
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 16,
})  


def custom_layout(n_plots, cols, fig_kwargs={}):
    rows = math.ceil(n_plots / cols)

    axs = np.full((rows, cols), None)
    fig = plt.figure(**fig_kwargs)
    gs = GridSpec(rows, cols * 2)

    full_rows = n_plots // cols
    extra = cols - n_plots % cols

    for i in range(n_plots):
        r, c = divmod(i, cols)
        if r < full_rows:
            axs[r, c] = fig.add_subplot(gs[r, 2 * c : 2 * (c + 1)])
        else:
            axs[r, c] = fig.add_subplot(gs[r, 2 * c + extra : 2 * (c + 1) + extra])
    return fig, axs


def plot_all_algorithms_convergence(
    algorithms,
    visuals_path,
    add_marker=False,
    show=False,
    save=True,
    show_time_memory=False,
    results=False,
):
    if show_time_memory:
        _, axs = custom_layout(5, 3, fig_kwargs={"figsize": (25, 12)})
        top_axs = axs[0]
    else:
        _, axs = plt.subplots(1, 3, figsize=(18, 5))
        top_axs = axs
    algo_plot_names = []

    # Find the maximum number of iterations among all algorithms
    n_iterations = max(algo.max_iterations for algo in algorithms)

    for algo in algorithms:
        label_name = algo.algo_plot_name
        algo_plot_names.append(label_name)
        for ax, values, label in zip(
            top_axs,
            [algo.mse_values, algo.mae_values, algo.f_values],
            ["MSE", "MAE", "Objective function"],
        ):
            iters = len(values)
            p = ax.plot(
                range(iters),
                values,
                label=label_name,
                marker="o" if add_marker else None,
                markersize=4 if add_marker else None,
            )
            ax.set_xlabel("Iteration")
            ax.set_ylabel(label)
            if label == "Objective function":
                ax.set_yscale("log")
            ax.grid(True)

            # Draw a cross marker if the algorithm stopped before reaching max_iterations
            if iters < algo.max_iterations:
                ax.plot(
                    iters - 1,
                    values[-1],
                    marker="x",
                    color="black",
                    markersize=10,
                )
            # Draw a dotted line for constant continuation to max_iterations
            if iters < n_iterations:
                ax.plot(
                    range(iters - 1, n_iterations),
                    [values[-1]] * (n_iterations - iters + 1),
                    linestyle=":",
                    color=p[0].get_color(),
                )
    if results:
        top_axs[2].legend(
            loc="upper right",
            fontsize=14,
        )
    else:
        top_axs[2].legend(
            bbox_to_anchor=(1.5, 0.95),
            title="Parameters",
        )

    if show_time_memory:
        cv_times = [algo.cv_time for algo in algorithms]
        memory_used_kb = [algo.memory_used / 1024 for algo in algorithms]
        axs[1, 0].barh(algo_plot_names, cv_times, color="skyblue")
        axs[1, 0].set_xlabel("Execution Time (s)")
        axs[1, 1].barh(algo_plot_names, memory_used_kb, color="lightgreen")
        axs[1, 1].set_xlabel("Peak Memory Used (KB)")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save:
        plt.savefig(os.path.join(visuals_path, "All_algorithms.pdf"))
    if show:
        plt.show()
    plt.close()


def plot_objective_functions_by_algorithm(list_of_algo_lists, visuals_path, add_marker=False, show=False, save=True):
    fig, axs = plt.subplots(1, len(list_of_algo_lists), figsize=(18, 5))
    max_iterations = max(algo.max_iterations for algo_list in list_of_algo_lists for algo in algo_list)
    for idx, algo_list in enumerate(list_of_algo_lists):
        for algo in algo_list:
            label_name = algo.algo_plot_name
            values = algo.f_values
            iters = len(values)
            p = axs[idx].plot(
                range(iters),
                values,
                label=label_name,
                marker="o" if add_marker else None,
                markersize=4 if add_marker else None,
            )

            # Draw a cross marker if the algorithm stopped before reaching max_iterations
            if iters < algo.max_iterations:
                axs[idx].plot(
                    iters - 1,
                    values[-1],
                    marker="x",
                    color="black",
                    markersize=10,
                )
            
            # Draw a dotted line for constant continuation to max_iterations
            if iters < max_iterations:
                axs[idx].plot(
                    range(iters - 1, max_iterations),
                    [values[-1]] * (max_iterations - iters + 1),
                    linestyle=":",
                    color=p[0].get_color(),
                )
        axs[idx].set_xlabel("Iteration")
        axs[idx].set_ylabel("Objective function")
        axs[idx].set_yscale("log")
        axs[idx].legend(loc="upper right", fontsize=14)
        axs[idx].grid(True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save:
        plt.savefig(os.path.join(visuals_path, "Objective_by_algorithm.pdf"))
    if show:
        plt.show()
