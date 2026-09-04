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


def plot_contrast_on_mesh(
    fields,
    vertices,
    triangles,
    titles=None,
    part="real",
    visuals_path=None,
    filename="contrast_on_mesh.pdf",
    show=False,
    save=True,
    cmap="viridis",
    share_scale=True,
):
    """Draw P0 (piecewise-constant) contrast vectors on the actual annulus.

    Until the mesh connectivity was exported (`savemesh` in
    scripts/GenerateMatrix*.edp) the comparison notebook could only plot
    `Re(m)` against the contrast dof index, because the coordinates were not
    available. `mesh.msh` supplies them, so a P0 field is exactly a per-face
    colour: `tripcolor(..., facecolors=values)` on the real geometry, as in
    the internship report's spatial iso-value plots.

    Args:
        fields: one contrast vector, or a sequence of them (each length P).
        vertices, triangles: as returned by `iwp.utils.mesh.read_freefem_mesh`.
        part: "real", "imag" or "abs".
        share_scale: put every panel on a common colour scale, so panels are
            comparable to each other rather than each self-normalized.
    """
    from iwp.utils.mesh import make_triangulation

    if isinstance(fields, np.ndarray) and fields.ndim == 1:
        fields = [fields]
    fields = list(fields)
    titles = list(titles) if titles is not None else [None] * len(fields)
    take = {"real": np.real, "imag": np.imag, "abs": np.abs}[part]
    values = [np.asarray(take(f), dtype=float) for f in fields]
    for v in values:
        if v.shape[0] != triangles.shape[0]:
            raise ValueError(
                f"field has {v.shape[0]} entries but the mesh has "
                f"{triangles.shape[0]} triangles: this is a P0 (per-face) plot."
            )

    tri = make_triangulation(vertices, triangles)
    vmin = min(v.min() for v in values) if share_scale else None
    vmax = max(v.max() for v in values) if share_scale else None

    fig, axs = plt.subplots(
        1, len(values), figsize=(5.2 * len(values), 4.6), squeeze=False
    )
    for ax, v, title in zip(axs[0], values, titles):
        tpc = ax.tripcolor(tri, facecolors=v, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        if title:
            ax.set_title(title)
        fig.colorbar(tpc, ax=ax, shrink=0.85)
    fig.tight_layout()
    if save and visuals_path is not None:
        fig.savefig(os.path.join(visuals_path, filename))
    if show:
        plt.show()
    return fig, axs[0]
