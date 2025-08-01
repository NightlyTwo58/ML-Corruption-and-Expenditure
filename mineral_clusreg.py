import project_1v2_0

all_exports_capita = project_1v2_0.load_pop_data()
labels = ['Cereals', 'Inorganic', 'Mineral', 'Ores', 'Wood']

minfilter = 1000
maxfilter = 50000
yscalar = 3
all_exports_capita[2] = project_1v2_0.remove_outliers(all_exports_capita[2], "dollar_per_capita", minfilter, maxfilter)
clusterdata = project_1v2_0.perform_kmeans_clustering(all_exports_capita[2], ['dollar_per_capita', 'HDI_value'], 5, yscalar)
clusterdata = project_1v2_0.country_code_to_names(clusterdata)
clusterdata.to_csv('data/clustering_results/mineral.csv', index=False)

project_1v2_0.combined_regression_clustering(
    [minfilter, maxfilter, yscalar],
    'Oil', full_clustered_df=clusterdata,
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
#     project_1v2_0.combined_regression_clustering(
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
