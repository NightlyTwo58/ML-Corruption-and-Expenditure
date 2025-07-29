import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.ticker as ticker
import seaborn as sns
import project_1v2_0

all_exports_capita = project_1v2_0.load_pop_data()
labels = ['Cereals', 'Inorganic', 'Mineral', 'Ores', 'Wood']


# project_1v2_0.plot_scatter_3d(all_exports_capita[2], 'Mineral')
# project_1v2_0.all_plot_scatter_3d(all_exports_capita, labels)
#
# plt.figure()
# project_1v2_0.kmeans_visual(all_exports_capita[2], ['dollar_per_capita', 'HDI_value'], 5, labels[2])
#
# project_1v2_0.perform_nonlinear_regression(
#     all_exports_capita[2], labels[2],
#     'dollar_per_capita',
#     'HDI_value',
#     model_func=project_1v2_0.power_law,
#     p0=[0.1, 0.1]
# )
# plt.show()


def remove_minoutliers(df, minchar, min_value):
    """
    Removes rows from a DataFrame where the value in a specified column
    is less than a given minimum value.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        minchar (str): The name of the column to check for the minimum value.
        min_value (float or int): The minimum threshold. Rows with values
                                  in 'minchar' less than this will be removed.

    Returns:
        pandas.DataFrame: A new DataFrame with rows removed based on the condition.
                          Returns a copy to avoid modifying the original DataFrame
                          in place.
    """
    print(f"Original DataFrame shape: {df.shape}")
    # df[minchar] = pd.to_numeric(df[minchar], errors='coerce')
    df = df.dropna(subset=[minchar])
    df = df.loc[df[minchar] >= min_value]
    print(f"DataFrame shape after filtering '{minchar}' less than {min_value}: {df.shape}")

    return df

def remove_maxoutliers(df, maxchar, max_value):
    """
    Removes rows from a DataFrame where the value in a specified column
    is larger than a given minimum value.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        maxchar (str): The name of the column to check for the minimum value.
        max_value (float or int): The max threshold. Rows with values
                                  in 'maxchar' larger than this will be removed.

    Returns:
        pandas.DataFrame: A new DataFrame with rows removed based on the condition.
                          Returns a copy to avoid modifying the original DataFrame
                          in place.
    """
    print(f"Original DataFrame shape: {df.shape}")
    # df[maxchar] = pd.to_numeric(df[maxchar], errors='coerce')
    df = df.dropna(subset=[maxchar])
    df = df.loc[df[maxchar] <= max_value]
    print(f"DataFrame shape after filtering '{maxchar}' less than {max_value}: {df.shape}")

    return df

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


mineral_mincap = remove_minoutliers(all_exports_capita[2], "dollar_per_capita", 1000)
mineral_cap = remove_maxoutliers(mineral_mincap, "dollar_per_capita", 50000)
# plt.figure()
# project_1v2_0.kmeans_visual(mineral_cap, ['dollar_per_capita', 'HDI_value'], 5, labels[2], 3)
# plt.show()

clusterdata = project_1v2_0.perform_kmeans_clustering(mineral_cap, ['dollar_per_capita', 'HDI_value'], 5, 3)
cluster0data = clusterdata[clusterdata['cluster'] != 0]
cluster23data = clusterdata[~clusterdata['cluster'].isin([2, 3])]
project_1v2_0.perform_nonlinear_regression(
            cluster0data.drop(columns='cluster'), labels[2],
            'dollar_per_capita',
            'HDI_value',
            model_func=project_1v2_0.power_law,
            p0=[0.1, 0.1]
        )
project_1v2_0.perform_nonlinear_regression(
            cluster23data.drop(columns='cluster'), labels[2],
            'dollar_per_capita',
            'HDI_value',
            model_func=project_1v2_0.power_law,
            p0=[0.1, 0.1]
        )
project_1v2_0.perform_linear_regression(cluster23data.drop(columns='cluster'), labels[2], ['dollar_per_capita', 'HDI_value'], 0.1)
project_1v2_0.kmeans_display(cluster0data, ['dollar_per_capita', 'HDI_value'], 5, labels[2], 3)


# mineral_curse = filter_df_by_cluster(mineral_mincap, ["dollar_per_capita", 'HDI_value'], 5, cluster_id_to_remove)