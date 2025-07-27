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

plt.figure()
project_1v2_0.kmeans_visual(all_exports_capita[2], ['dollar_per_capita', 'HDI_value'], 5, labels[2])

project_1v2_0.perform_nonlinear_regression(
    all_exports_capita[2], labels[2],
    'dollar_per_capita',
    'HDI_value',
    model_func=project_1v2_0.power_law,
    p0=[0.1, 0.1]
)
plt.show()