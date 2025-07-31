import project_1v2_0

all_exports_capita = project_1v2_0.load_pop_data()

resource = all_exports_capita[4]
minfilter = 100
maxfilter = None
yscalar = 1
resource = project_1v2_0.remove_outliers(resource, "dollar_per_capita", minfilter, maxfilter)
clusterdata = project_1v2_0.perform_kmeans_clustering(resource, ['dollar_per_capita', 'HDI_value'], 5, yscalar)

project_1v2_0.combined_regression_clustering(
    [minfilter, maxfilter, yscalar],
    'Wood', full_clustered_df=clusterdata,
    cluster_filtered_dfs=[
        # (clusterdata, [1, 5, 4, 3], 'C1'),
        (clusterdata, [3, 4], 'C2'),
        (clusterdata, [4], 'C3')
    ],
    xcol='dollar_per_capita',
    ycol='HDI_value',
    model_func=project_1v2_0.power_law,
    p0=[0.1, 0.1]
)