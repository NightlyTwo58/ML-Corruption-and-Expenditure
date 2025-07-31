import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from src import project_1v2_0

all_exports_capita = project_1v2_0.load_pop_data()
labels = ['Cereals', 'Inorganic', 'Mineral', 'Ores', 'Wood']

def filter_df_by_cluster(df, features, n_clusters, cluster_id_to_remove):
    """
    Performs K-Means clustering on the DataFrame and then returns a new DataFrame
    with a specified cluster removed.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        features (list): A list of feature (column) names to use for clustering.
        n_clusters (int): The number of clusters to form.
        cluster_id_to_remove (int): The integer ID of the cluster to remove from the DataFrame.

    Returns:
        pandas.DataFrame: A new DataFrame with the specified cluster removed.
                          Returns a copy to avoid modifying the original DataFrame.
    """
    df_clustered, centroids = project_1v2_0.perform_kmeans_clustering(df, features, n_clusters)
    df_filtered = df_clustered[df_clustered['cluster'] != cluster_id_to_remove].copy()
    df_final = df_filtered.drop(columns=['cluster'])
    return df_final

def fit_and_plot_powerlaw(ax, df, xcol, ycol, removed_clusters, scatter_label=True, model_func=None, p0=None):
    """
    Fits model_func to (xcol, ycol) of df, plots scatter and fitted curve.
    removed_clusters: iterable of ints that were removed (for label text).
    """
    x = df[xcol].to_numpy()
    y = df[ycol].to_numpy()

    # Filter out non-positive if power law requires it (optional; depends on your data)
    valid = (~np.isnan(x)) & (~np.isnan(y)) & (x > 0) & (y > 0)
    x = x[valid]
    y = y[valid]
    if len(x) == 0:
        raise ValueError("No valid data for fitting after filtering non-positive or NaNs.")

    # Fit
    popt, pcov = curve_fit(model_func, x, y, p0=p0, maxfev=10000)

    # Scatter: only label once if desired
    if scatter_label:
        label_scatter = f"Data (removed clusters: {','.join(map(str, removed_clusters)) or 'none'})"
    else:
        label_scatter = None  # no label to avoid duplicates

    ax.scatter(x, y, label=label_scatter, alpha=0.6, edgecolors='none')

    # Regression curve: sort for smooth line
    x_fit = np.linspace(np.min(x), np.max(x), 300)
    y_fit = model_func(x_fit, *popt)
    label_curve = f"Fit (removed: {','.join(map(str, removed_clusters)) or 'none'})"
    ax.plot(x_fit, y_fit, label=label_curve, linewidth=2)

    a, b = popt
    param_text = f"a={a:.3g}, b={b:.3g}"
    ax.text(0.05, 0.95, param_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', alpha=0.2))

mineral_filtered = project_1v2_0.remove_outliers(all_exports_capita[2], "dollar_per_capita", 1000, 50000)

clusterdata = project_1v2_0.perform_kmeans_clustering(mineral_filtered, ['dollar_per_capita', 'HDI_value'], 5, 3)
cluster0data = clusterdata[clusterdata['cluster'] != 0]
cluster23data = clusterdata[~clusterdata['cluster'].isin([2, 3])]
# project_1v2_0.perform_nonlinear_regression(
#             cluster0data.drop(columns='cluster'), labels[2],
#             'dollar_per_capita',
#             'HDI_value',
#             model_func=project_1v2_0.power_law,
#             p0=[0.1, 0.1]
#         )
# project_1v2_0.perform_nonlinear_regression(
#             cluster23data.drop(columns='cluster'), labels[2],
#             'dollar_per_capita',
#             'HDI_value',
#             model_func=project_1v2_0.power_law,
#             p0=[0.1, 0.1]
#         )
# project_1v2_0.perform_linear_regression(cluster23data.drop(columns='cluster'), labels[2], ['dollar_per_capita', 'HDI_value'], 0.1)
# project_1v2_0.kmeans_display(cluster0data, ['dollar_per_capita', 'HDI_value'], 5, labels[2], 3)

fig, ax = plt.subplots(figsize=(7,5))

fit_and_plot_powerlaw(
    ax,
    cluster0data.drop(columns='cluster'),
    xcol='dollar_per_capita',
    ycol='HDI_value',
    removed_clusters=[0],
    model_func=project_1v2_0.power_law,
    p0=[0.1, 0.1]
)

fit_and_plot_powerlaw(
    ax,
    cluster23data.drop(columns='cluster'),
    xcol='dollar_per_capita',
    ycol='HDI_value',
    removed_clusters=[2, 3],
    scatter_label=False,
    model_func=project_1v2_0.power_law,
    p0=[0.1, 0.1]
)

# ax.set_xscale('log')
# ax.set_yscale('log')
ax.set_xlabel('dollar_per_capita')
ax.set_ylabel('HDI_value')
ax.set_title(f"Nonlinear regression with clusters removed")
project_1v2_0.dedupe_legend(ax)
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()
