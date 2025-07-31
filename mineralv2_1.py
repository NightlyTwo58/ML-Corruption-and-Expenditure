import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import project_1v2_0

def fit_powerlaw(x, y, model_func, p0):
    popt, _ = curve_fit(model_func, x, y, p0=p0, maxfev=10000)
    y_pred = model_func(x, *popt)
    r2 = r2_score(y, y_pred)
    return popt, r2

def plot_with_clustered_data_and_fits(filteringBounds, name, full_clustered_df,
                                      cluster_filtered_dfs,
                                      xcol, ycol,
                                      model_func,
                                      p0,
                                      figsize=(8,5)):
    """
    full_clustered_df: DataFrame including 'cluster' column (the original KMeans result)
    cluster_filtered_dfs: list of tuples: (df_after_removal_with_no_cluster_col, removed_clusters_list, fit_color)
    """
    n_clusters = int(full_clustered_df['cluster'].max() + 1)
    palette = sns.color_palette("tab10", n_colors=n_clusters)

    fig, ax = plt.subplots(figsize=figsize)

    for ci in range(n_clusters):
        subset = full_clustered_df[full_clustered_df['cluster'] == ci]
        if subset.empty:
            continue
        x = subset[xcol].to_numpy()
        y = subset[ycol].to_numpy()
        ax.scatter(x, y, label=f"Cluster {ci}", alpha=0.6, edgecolors='none', s=40,
                   color=palette[ci])

    # Fit overlays: for each filtered dataset (which has had some clusters removed),
    # fit and draw the nonlinear curve, annotate R^2 and removed clusters.
    for df_filtered, removed_clusters, fit_color in cluster_filtered_dfs:
        df_filtered = df_filtered[~df_filtered['cluster'].isin(removed_clusters)].drop(columns='cluster')

        x = df_filtered[xcol].to_numpy()
        y = df_filtered[ycol].to_numpy()

        popt, r2 = fit_powerlaw(x, y, model_func, p0=p0)

        # Plot fit curve: use a dense sorted x for smoothness
        x_fit = np.linspace(np.min(x), np.max(x), 300)
        y_fit = model_func(x_fit, *popt)
        removed_str = ",".join(str(c) for c in removed_clusters) if removed_clusters else "none"
        label_curve = f"Fit (removed: {removed_str})"

        ax.plot(x_fit, y_fit, label=label_curve, linewidth=2, color=fit_color or None)

        # Place R² near the right end of the curve
        x_annot = x_fit[-1]
        y_annot = y_fit[-1]
        ax.annotate(
            rf"$R^2$={r2:.3f}",
            xy=(x_annot, y_annot),
            xytext=(5, 5 + (5 if removed_str != "none" else 0)),
            textcoords="offset points",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", alpha=0.3),
            arrowprops=dict(arrowstyle="->", lw=0.5)
        )

    # ax.set_xscale('log')
    # ax.set_yscale('log')
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.set_title(f"Scaled Nonlinear Regression with Clusters Removed for {name}\nFiltered data between {filteringBounds[0]}, {filteringBounds[1]}, {filteringBounds[2]}:1 horizontal scaling", fontsize=12)

    # Deduplicate legend: keep one entry per label
    handles, labels = ax.get_legend_handles_labels()
    by_label = {}
    for h, l in zip(handles, labels):
        if l not in by_label:
            by_label[l] = h
    ax.legend(by_label.values(), by_label.keys(), frameon=True, fontsize=8, loc='lower right')

    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

all_exports_capita = project_1v2_0.load_pop_data()
labels = ['Cereals', 'Inorganic', 'Mineral', 'Ores', 'Wood']

minfilter = 1000
maxfiler = 50000
yscalar = 3
all_exports_capita[2] = project_1v2_0.remove_outliers(all_exports_capita[2], "dollar_per_capita", minfilter, maxfiler)
clusterdata = project_1v2_0.perform_kmeans_clustering(all_exports_capita[2], ['dollar_per_capita', 'HDI_value'], 5, 3)

plot_with_clustered_data_and_fits(
    [minfilter, maxfiler, yscalar],
    labels[2], full_clustered_df=clusterdata,
    cluster_filtered_dfs=[
        (clusterdata, [0], 'C1'),
        (clusterdata, [2, 3], 'C2')
    ],
    xcol='dollar_per_capita',
    ycol='HDI_value',
    model_func=project_1v2_0.power_law,
    p0=[0.1, 0.1]
)

# TODO: uniform projections for all. However, as we lack an autoscaling function and a automated pattern detection
#  function, this results in inconsistency.
# for i in range(0, 5):
#     all_exports_capita[i] = project_1v2_0.remove_outliers(all_exports_capita[i], "dollar_per_capita", 1000, "dollar_per_capita", 50000)
#     clusterdata = project_1v2_0.perform_kmeans_clustering(all_exports_capita[i], ['dollar_per_capita', 'HDI_value'], 5, 3)
#
#     fit_colors = ['C1', 'C2']
#
#     plot_with_clustered_data_and_fits(
#         labels[i], full_clustered_df=clusterdata,
#         cluster_filtered_dfs=[
#             (clusterdata, [0], fit_colors[0]),
#             (clusterdata, [2, 3], fit_colors[1])
#         ],
#         xcol='dollar_per_capita',
#         ycol='HDI_value',
#         model_func=project_1v2_0.power_law,
#         p0=[0.1, 0.1]
#     )
