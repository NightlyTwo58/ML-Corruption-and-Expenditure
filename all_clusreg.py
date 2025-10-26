import numpy as np
import pandas as pd

import project_1v3

def model_func(x, a, b): return a * np.log(b * x + 1)

cereals = pd.read_csv("data/Exports Per Capita/Cereals_capita.csv")
inorganic = pd.read_csv("data/Exports Per Capita/Inorganic_capita.csv")
mineral = pd.read_csv("data/Exports Per Capita/Mineral_capita.csv")
ores = pd.read_csv("data/Exports Per Capita/Ores_capita.csv")
wood = pd.read_csv("data/Exports Per Capita/Wood_capita.csv")
folder = "data/auto_clustering/"
p0 = [1, 0.7]

cereal_df, cereal_km = project_1v3.combined_NN_clustering(
    df=cereals,
    name="Cereal Exports p/c vs HDI",
    xcol='dollar_per_capita',
    ycol='HDI_value',
    filteringBounds=(10, 400, 1),
    n_clusters=4,
    cluster_filtered_dfs=[
        # ([2], 'C1'),
        ([], 'C2')
    ]
)

project_1v3.save_clustering_results(cereal_df, cereal_km, 'cereal', folder)

mineral_df, mineral_km = project_1v3.combined_NN_clustering(
    df=mineral,
    name="Mineral Exports p/c vs HDI",
    xcol='dollar_per_capita',
    ycol='HDI_value',
    filteringBounds=(1000, 50000, 3),
    n_clusters=5,
    cluster_filtered_dfs=[
        # ([0], 'C1'),
        # ([2, 3], 'C2'),
        ([], 'C3'),
    ]
)

project_1v3.save_clustering_results(mineral_df, mineral_km, 'mineral', folder)

inorganic_df, inorganic_km = project_1v3.combined_NN_clustering(
    df=inorganic,
    name="Inorganic Exports p/c vs HDI",
    xcol='dollar_per_capita',
    ycol='HDI_value',
    filteringBounds=(150, None, 2),
    n_clusters=5,
    cluster_filtered_dfs=[
        ([], 'C1'),
        # ([1], 'C2')
    ]
)

project_1v3.save_clustering_results(inorganic_df, inorganic_km, 'inorganic', folder)

ores_df, ores_km = project_1v3.combined_NN_clustering(
    df=ores,
    name="Ore Exports p/c vs HDI",
    xcol='dollar_per_capita',
    ycol='HDI_value',
    filteringBounds=(100, None, 1),
    n_clusters=7,
    cluster_filtered_dfs=[
        # ([1, 5, 4, 3], 'C1'),
        # ([6], 'C2'),
        ([], 'C3')
    ],
)

project_1v3.save_clustering_results(ores_df, ores_km, 'ores', folder)

wood_df, wood_km = project_1v3.combined_NN_clustering(
    df=wood,
    name="Wood Exports p/c vs HDI",
    xcol='dollar_per_capita',
    ycol='HDI_value',
    filteringBounds=(100, None, 1),
    n_clusters=5,
    cluster_filtered_dfs=[
        # ([3, 4], 'C1'),
        # ([4], 'C2'),
        ([], 'C3')
    ],
)

project_1v3.save_clustering_results(wood_df, wood_km, 'wood', folder)

