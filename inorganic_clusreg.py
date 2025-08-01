import project_1v2_0

all_exports_capita = project_1v2_0.load_pop_data()

minfilter = 150
maxfilter = None
yscalar = 2
all_exports_capita[1] = project_1v2_0.remove_outliers(all_exports_capita[1], "dollar_per_capita", minfilter, maxfilter)
clusterdata = project_1v2_0.perform_kmeans_clustering(all_exports_capita[1], ['dollar_per_capita', 'HDI_value'], 5, yscalar)
clusterdata = project_1v2_0.country_code_to_names(clusterdata)
clusterdata.to_csv('data/clustering_results/inorganic.csv', index=False)

project_1v2_0.combined_regression_clustering(
    [minfilter, maxfilter, yscalar],
    'Rare Earth Metals', full_clustered_df=clusterdata,
    cluster_filtered_dfs=[
        # (clusterdata, [1, 5, 4, 3], 'C1'),
        (clusterdata, [], 'C2'),
        (clusterdata, [1], 'C3')
    ],
    xcol='dollar_per_capita',
    ycol='HDI_value',
    model_func=project_1v2_0.power_law,
    p0=[0.1, 0.1]
)