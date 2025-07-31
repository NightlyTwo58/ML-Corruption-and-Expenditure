import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
from scipy.stats import linregress
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


def load_data():
    """
    Loads various natural resource export datasets and stores them in a list.

    Returns:
        list: A list containing pandas DataFrames for cereals, inorganic,
              mineral, ores, and wood exports.
    """
    cereals = pd.read_csv("data/Exports Data Comb/RCereals.csv_with_HDI.csv")
    inorganic = pd.read_csv("data/Exports Data Comb/RInorganic.csv_with_HDI.csv")
    mineral = pd.read_csv("data/Exports Data Comb/RMineral.csv_with_HDI.csv")
    ores = pd.read_csv("data/Exports Data Comb/ROres.csv_with_HDI.csv")
    wood = pd.read_csv("data/Exports Data Comb/RWood.csv_with_HDI.csv")
    return [cereals, inorganic, mineral, ores, wood]

def load_pop_data():
    cereals = pd.read_csv("data/Exports Per Capita/Cereals_capita.csv")
    inorganic = pd.read_csv("data/Exports Per Capita/Inorganic_capita.csv")
    mineral = pd.read_csv("data/Exports Per Capita/Mineral_capita.csv")
    ores = pd.read_csv("data/Exports Per Capita/Ores_capita.csv")
    wood = pd.read_csv("data/Exports Per Capita/Wood_capita.csv")
    return [cereals, inorganic, mineral, ores, wood]

def analyze_hdi_correlation(all_exports, labels):
    """
    Calculates and prints the correlation between 'dollar_value' and 'HDI_value'
    for each resource export dataset.

    Args:
        all_exports (list): A list of pandas DataFrames, each representing
                            a different natural resource export.
        labels (list): A list of strings, where each string is the name
                       corresponding to a DataFrame in all_exports.
    """
    correlation_df = pd.DataFrame(index=labels, columns=['HDI Correlation'])
    for data, label in zip(all_exports, labels):
        correlation = data['dollar_per_capita'].astype(float).corr(data['HDI_value'].astype(float))
        correlation_df.loc[label, 'HDI Correlation'] = correlation
    print(correlation_df.iloc[:, 0])

def plot_scatter(df, title_suffix):
    """
    Generates a scatter plot of 'dollar_value' vs 'HDI_value'.

    Args:
        df (pandas.DataFrame): The DataFrame to plot.
        title_suffix (str): A string to append to the plot titles.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(df['dollar_per_capita'], df['HDI_value'])
    plt.title(f'Exports Per Capita (Dollars) vs HDI for {title_suffix}')
    plt.xlabel('Exports Per Capita (Dollars)')
    plt.ylabel('HDI Value')
    plt.show()

def plot_scatter_3d(df, title_suffix):
    """
    Generates a 3D scatter plot of 'dollar_per_capita' vs 'HDI_value' vs 'year'.

    Args:
        df (pandas.DataFrame): The DataFrame to plot. Expected to have
                               'dollar_per_capita', 'HDI_value', and 'year' columns.
        title_suffix (str): A string to append to the plot titles.
    """
    required_cols = ["dollar_per_capita", "HDI_value", "year"]
    df_cleaned = df.dropna(subset=required_cols).copy()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(df_cleaned['dollar_per_capita'],
               df_cleaned['HDI_value'],
               df_cleaned['year'],
               c='blue',
               alpha=0.7,
               edgecolors='k')

    ax.set_xlabel('Exports Per Capita (Dollars)')
    ax.set_ylabel('HDI Value')
    ax.set_zlabel('Year')

    ax.set_title(f'3D Scatter Plot: Exports Per Capita vs HDI vs Year for {title_suffix}')

    plt.tight_layout()
    plt.show()

def all_plot_scatter(df, labels):
    """
    Generates a scatter plot of 'dollar_value' vs 'HDI_value' for multiple DataFrames.

    Args:
        df (list): A list of pandas.DataFrames, where each DataFrame
                            represents a different resource type.
        labels (list): A list of strings, where each string is the name
                       corresponding to a DataFrame in list_of_dfs.
    """
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'black', 'gray']

    plt.figure(figsize=(12, 6), dpi=100)

    for i, data_df in enumerate(df):
        label = labels[i]
        data_df = data_df.dropna()
        plt.scatter(data_df["dollar_per_capita"], data_df["HDI_value"],
                    c=colors[i], label=label, alpha=0.7, edgecolors='k')

    plt.xlabel("Export Amount (Dollars) Per Capita")
    plt.ylabel("HDI Value")
    plt.title("Scatter Plot of Different Resource Types")
    plt.legend()
    plt.grid(True)
    plt.show()

def all_plot_scatter_3d(list_of_dfs, labels):
    """
    Generates a 3D scatter plot of 'dollar_per_capita' vs 'HDI_value' vs 'year'
    for multiple DataFrames.

    Args:
        list_of_dfs (list): A list of pandas.DataFrames, where each DataFrame
                            represents a different resource type. Each DataFrame
                            is expected to have 'dollar_per_capita', 'HDI_value',
                            and 'year' columns.
        labels (list): A list of strings, where each string is the name
                       corresponding to a DataFrame in list_of_dfs.
    """
    if not isinstance(list_of_dfs, list) or not all(isinstance(d, pd.DataFrame) for d in list_of_dfs):
        raise TypeError("list_of_dfs must be a list of pandas DataFrames.")
    if not isinstance(labels, list) or not all(isinstance(l, str) for l in labels):
        raise TypeError("labels must be a list of strings.")
    if len(list_of_dfs) != len(labels):
        raise ValueError("The number of DataFrames and labels must be the same.")

    colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'black', 'gray']

    fig = plt.figure(figsize=(12, 10), dpi=100)
    ax = fig.add_subplot(111, projection='3d')

    for i, data_df in enumerate(list_of_dfs):
        color = colors[i % len(colors)]
        label = labels[i]

        required_cols = ["dollar_per_capita", "HDI_value", "year"]
        data_cleaned = data_df.dropna(subset=required_cols).copy()

        ax.scatter(data_cleaned["dollar_per_capita"],
                   data_cleaned["HDI_value"],
                   data_cleaned["year"],
                   c=color, label=label, alpha=0.7, edgecolors='k')

    ax.set_xlabel("Export Amount (Dollars) Per Capita")
    ax.set_ylabel("HDI Value")
    ax.set_zlabel("Year")

    ax.set_title("3D Scatter Plot of Different Resource Types")
    ax.legend()
    plt.tight_layout()
    plt.show()


def histogram(df):
    """
    Generates a bar chart of the yearly average 'HDI_value' for a given DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing 'year' and 'HDI_value' data.
    """
    df['HDI_value'] = pd.to_numeric(df['HDI_value'], errors='coerce')
    df_cleaned = df.dropna(subset=['HDI_value'])
    yearly_avg_hdi = df_cleaned.groupby('year')['HDI_value'].mean().reset_index()

    plt.figure(figsize=(12, 7))
    plt.bar(yearly_avg_hdi['year'], yearly_avg_hdi['HDI_value'], color='skyblue')

    plt.title('Yearly Average HDI Value')
    plt.xlabel('Year')
    plt.ylabel('Average HDI Value')
    plt.xticks(yearly_avg_hdi['year'].astype(int))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def perform_kmeans_clustering_auto(df, features, n_clusters):
    """
    Performs K-Means clustering on the given DataFrame and features after Min-Max scaling.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        features (list): A list of feature (column) names to use for clustering.
                         Expected to be 2 features for typical 2D visualization contexts.
        n_clusters (int): The number of clusters to form.

    Returns:
        tuple: A tuple containing:
            - pandas.DataFrame: The original DataFrame (cleaned of NaNs in features)
                                with a new 'cluster' column indicating assignment.
            - numpy.ndarray: The cluster centroids in the original (unscaled) feature space.
    """
    df_cleaned = df.dropna(subset=features).copy()

    scaler = MinMaxScaler()
    X_original = df_cleaned[features]
    X_scaled = pd.DataFrame(scaler.fit_transform(X_original), columns=features, index=X_original.index)

    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
    df_cleaned['cluster'] = kmeans.fit_predict(X_scaled)  # Assign clusters to the cleaned DataFrame

    centroids_scaled = kmeans.cluster_centers_
    centroids_original_scale = scaler.inverse_transform(centroids_scaled)

    return df_cleaned, centroids_original_scale


def perform_kmeans_clustering(df, features, n_clusters, y_scalar):
    """
    Performs K-Means clustering on the given DataFrame using scaled features with an adjustable y-axis scaling factor,
    and returns the cleaned DataFrame with a new 'cluster' column.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        features (list): A list of two feature names [x, y] to use for clustering.
        n_clusters (int): The number of clusters to form.
        y_scalar (float): Factor to scale the y-feature relative to x-feature for clustering sensitivity.

    Returns:
        pandas.DataFrame: The cleaned DataFrame with a new 'cluster' column added.
    """
    df_cleaned = df.dropna(subset=features).copy()
    X_original = df_cleaned[features]

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    x_scaled = scaler_x.fit_transform(X_original[[features[0]]])
    y_scaled = scaler_y.fit_transform(X_original[[features[1]]]) * y_scalar

    X_scaled_custom = np.hstack((x_scaled, y_scaled))

    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
    df_cleaned['cluster'] = kmeans.fit_predict(X_scaled_custom)

    return df_cleaned

def kmeans_display(df, features, n_clusters, name, y_scalar):
    sns.scatterplot(data=df, x=features[0], y=features[1], hue='cluster', palette='viridis', s=50)

    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.title(f"K-Means Clustering for {name} (k={n_clusters}) with 1:{y_scalar} Scaled Clustering")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def kmeans_visual(df, features, n_clusters, name, y_scalar):
    """
    Performs K-Means clustering on specified features with Min-Max scaling,
    and visualizes the clusters and centroids.

    Args:
        df (pd.DataFrame): The input DataFrame.
        features (list): A list of two strings, representing the X and Y feature names
                         to be used for clustering and plotting.
        n_clusters (int): The number of clusters for K-Means.
        name {str): name of the export graphed
        y_scalar (int): the ratio of y-scaling (of HDI) to x. For example, 2 means clustering is twice as sensitive to HDI than exports.
    """
    df_cleaned = df.dropna(subset=features).copy()

    X_original = df_cleaned[features]
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    x_scaled = scaler_x.fit_transform(X_original[[features[0]]])
    y_scaled = scaler_y.fit_transform(X_original[[features[1]]]) * y_scalar

    X_scaled_custom = np.hstack((x_scaled, y_scaled))

    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
    df_cleaned['cluster'] = kmeans.fit_predict(X_scaled_custom)

    centroids_scaled = kmeans.cluster_centers_

    centroids_x = scaler_x.inverse_transform(centroids_scaled[:, [0]])
    centroids_y = scaler_y.inverse_transform(centroids_scaled[:, [1]] / y_scalar)
    centroids_original = np.hstack((centroids_x, centroids_y))

    sns.scatterplot(data=df_cleaned, x=features[0], y=features[1], hue='cluster', palette='viridis', s=50)

    plt.scatter(centroids_original[:, 0], centroids_original[:, 1],
                c='black', s=100, marker='x', label='Centroids')

    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.title(f"K-Means Clustering for {name} (k={n_clusters}) with 1:{y_scalar} Scaled Clustering")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def perform_linear_regression(df, label, features, y_scalar):
    """
    Performs linear regression with optional y-axis scaling and plots the result
    in original data coordinates to align with k-means and nonlinear plots.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        label (str): Title/label for the plot.
        features (list[str]): [x_feature, y_feature].
        y_scalar (float): Relative scaling factor applied to the y-feature before fitting
                          (e.g., 0.5 to de-emphasize y-feature in regression).
    """
    x_feature, y_feature = features
    df_filtered = df[[x_feature, y_feature]].dropna()

    if df_filtered.empty:
        print(f"Warning: No valid data points for linear regression after dropping NaNs for {x_feature} and {y_feature}.")
        return

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    x_scaled = scaler_x.fit_transform(df_filtered[[x_feature]])
    y_scaled = scaler_y.fit_transform(df_filtered[[y_feature]]) * y_scalar

    x_flat = x_scaled.flatten()
    y_flat = y_scaled.flatten()

    slope, intercept, r_value, p_value, std_err = linregress(x_flat, y_flat)
    y_pred_scaled = slope * x_flat + intercept

    x_original = df_filtered[x_feature].values.reshape(-1, 1)
    y_pred_unscaled = scaler_y.inverse_transform((y_pred_scaled / y_scalar).reshape(-1, 1)).flatten()

    plt.scatter(df_filtered[x_feature], df_filtered[y_feature], alpha=0.5, label="Scaled Data")

    plt.plot(df_filtered[x_feature], y_pred_unscaled, color='red',
             label=f"Linear Fit (R²={r_value**2:.4f})")

    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.xlabel(x_feature)
    plt.ylabel(y_feature)
    plt.title(f"Linear Regression: {y_feature} vs {x_feature} for {label}\n(Scaled {x_feature}:1, {y_feature}:{y_scalar})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    print(f"\nLinear Fit Parameters: slope={slope:.4f}, intercept={intercept:.4f}, R-squared={r_value**2:.4f}")


def perform_nonlinear_regression(df, label, feature, target, model_func, p0=None, segments=None):
    """
    Performs non-linear regression using curve_fit and plots the results.
    Can fit multiple segments if 'segments' are provided.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        feature (str): The name of the independent variable column.
        target (str): The name of the dependent variable column.
        model_func (callable): The non-linear function to fit (e.g., power_law).
                               It must take x as the first argument and then parameters.
        p0 (list, optional): Initial guess for the parameters of the model_func.
                             Defaults to None, letting curve_fit determine.
        segments (list of float, optional): A list of 'dollar_value' points to segment the data.
                                            If None, a single fit is performed.
                                            Example: [1e9, 5e9] to segment at those dollar values.
    """
    df_filtered = df[[feature, target]].dropna()

    if df_filtered.empty:
        print(f"Warning: No valid data points for non-linear regression after dropping NaNs for {feature} and {target}.")
        return

    x_data_full = df_filtered[feature].values
    y_data_full = df_filtered[target].values

    plt.scatter(x_data_full, y_data_full, alpha=0.5, label="Data")

    if segments:
        segment_points = sorted([min(x_data_full)] + segments + [max(x_data_full)])
        colors = plt.cm.viridis(np.linspace(0, 1, len(segment_points) - 1)) # Use a colormap for segments

        for i in range(len(segment_points) - 1):
            start = segment_points[i]
            end = segment_points[i+1]

            segment_df = df_filtered[(df_filtered[feature] >= start) & (df_filtered[feature] <= end)]
            x_segment = segment_df[feature].values
            y_segment = segment_df[target].values

            if len(x_segment) > (len(p0) if p0 is not None else 2):
                try:
                    params, covariance = curve_fit(model_func, x_segment, y_segment, p0=p0, maxfev=5000)
                    y_pred_segment = model_func(x_segment, *params)
                    ss_res = np.sum((y_segment - y_pred_segment) ** 2)
                    ss_tot = np.sum((y_segment - np.mean(y_segment)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan

                    x_fit_segment = np.linspace(start, end, 500)
                    y_fit_segment = model_func(x_fit_segment, *params)
                    plt.plot(x_fit_segment, y_fit_segment, color=colors[i],
                             label=f"Segment {i+1} Fit (R2={r_squared:.4f})", linestyle='--')
                    print(f"\nSegment {i+1} ({start:.2e} to {end:.2e}) Parameters: {params}, R-squared: {r_squared:.4f}")
                except RuntimeError as e:
                    print(f"Warning: Could not fit curve for segment {i+1} ({start:.2e} to {end:.2e}). Error: {e}")
            else:
                print(f"Warning: Not enough data points for fitting in segment {i+1} ({start:.2e} to {end:.2e}).")
    else:
        try:
            params, covariance = curve_fit(model_func, x_data_full, y_data_full, p0=p0, maxfev=5000)
            y_pred_full = model_func(x_data_full, *params)
            ss_res = np.sum((y_data_full - y_pred_full) ** 2)
            ss_tot = np.sum((y_data_full - np.mean(y_data_full)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan

            x_fit_full = np.linspace(min(x_data_full), max(x_data_full), 2000)
            y_fit_full = model_func(x_fit_full, *params)
            plt.plot(x_fit_full, y_fit_full, color='red', label=f"Fitted Curve (R2={r_squared:.4f})")
            print(f"\nOverall Fit Parameters: {params}, R-squared: {r_squared:.4f}")
        except RuntimeError as e:
            print(f"Warning: Could not fit curve for the entire dataset. Error: {e}")

    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1e10))
    plt.xlabel(feature)
    plt.ylabel(target)
    plt.title(f"Nonlinear Regression: {target} vs {feature} for {label} (Power Function Fit)")
    plt.legend()
    plt.grid(True)

def remove_outliers(df, col, min_value=None, max_value=None):
    """
    Removes rows based on optional minimum and/or maximum thresholds for specified columns.
    If a bound is not provided (None), it is skipped.

    Args:
        df (pandas.DataFrame): Input DataFrame.
        col (str): Column name for bound filtering.
        min_value (float or int or None): Minimum threshold; rows with df[minchar] < min_value are removed.
        max_value (float or int or None): Maximum threshold; rows with df[maxchar] > max_value are removed.

    Returns:
        pandas.DataFrame: Filtered copy of the DataFrame.
    """
    orig_shape = df.shape
    df_out = df.copy()

    if col is not None and min_value is not None:
        if col not in df_out.columns:
            raise KeyError(f"Column '{col}' not in DataFrame.")
        df_out = df_out.dropna(subset=[col])
        before = df_out.shape
        df_out = df_out[df_out[col].between(min_value, max_value, inclusive="both")]
        after = df_out.shape
        print(f"Applied filter on {min_value} <= '{col}' <= {max_value}: {before} -> {after}")

    if min_value is None or max_value is None:
        print(f"No filters applied; returning original shape {orig_shape}")

    return df_out

def power_law(x, a, b):
    """
    Defines a power-law function for non-linear regression.

    Args:
        x (numpy.ndarray): The independent variable.
        a (float): The scaling parameter.
        b (float): The exponent parameter.

    Returns:
        numpy.ndarray: The calculated y values.
    """
    # Added a small constant to prevent issues with x=0
    return a * np.power(x + 1e-9, b)

def dedupe_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), frameon=True)

if __name__ == "__main__":
    all_exports = load_data()
    all_exports_capita = load_pop_data()
    labels = ['Cereals', 'Inorganic', 'Mineral', 'Ores', 'Wood']

    print("--- Head of each DataFrame ---")
    for label, df in zip(labels, all_exports_capita):
        print(f"\n{label} Data Head:")
        print(df.head())

    print("\n--- Size of each DataFrame ---")
    for label, df in zip(labels, all_exports_capita):
        print(f"{label} Size: {df.size}")

    print("\n--- Plotting and HDI Correlation ---")
    histogram(all_exports_capita[0])

    all_plot_scatter(all_exports_capita, labels)

    for export, label in zip(all_exports_capita, labels):
        plot_scatter(export, label)

    print("\nHDI Correlation for each resource:")
    analyze_hdi_correlation(all_exports_capita, labels)

    for i, export in enumerate(all_exports_capita):
        plt.figure(i)
        kmeans_visual(export, ['dollar_per_capita', 'HDI_value'], 5, labels[i], 2)
    plt.show()

    print("\n--- Performing Non-linear Regression (Power Law) ---")

    for i, (export, label) in enumerate(zip(all_exports_capita, labels)):
        plt.figure(i)
        perform_nonlinear_regression(
            export, label,
            'dollar_per_capita',
            'HDI_value',
            model_func=power_law,
            p0=[0.1, 0.1]
        )

    plt.show()

    # with split functionality; currently, split functionality is x-income based
    # perform_nonlinear_regression(
    #     all_exports[1],
    #     'dollar_value',
    #     'HDI_value',
    #     model_func=power_law,
    #     p0=[0.1, 0.1],
    #     segments=[1e9, 5e9]
    # )