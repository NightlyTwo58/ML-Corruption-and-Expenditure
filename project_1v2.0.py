import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.ticker as ticker

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
        correlation = data['dollar_value'].astype(float).corr(data['HDI_value'].astype(float))
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
    plt.scatter(df['dollar_value'], df['HDI_value'])
    plt.title(f'Dollar_value vs HDI_value for {title_suffix}')
    plt.xlabel('Dollar Value')
    plt.ylabel('HDI Value')
    plt.show()

def histogram(df):
    """
    Generates a bar chart of the yearly average 'HDI_value' for a given DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing 'year' and 'HDI_value' data.
        title_suffix (str): A string to append to the plot title.
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

def perform_kmeans_clustering(df, features, n_clusters=3):
    """
    Performs K-Means clustering on the specified features of a DataFrame,
    handles NaNs by dropping rows, and prints cluster centroids.

    Args:
        df (pandas.DataFrame): The DataFrame to perform clustering on.
        features (list): A list of column names to be used for clustering.
        n_clusters (int): The number of clusters to form.

    Returns:
        tuple: A tuple containing:
            - pandas.DataFrame: The DataFrame with a new 'cluster' column.
            - numpy.ndarray: The cluster centroids.
    """
    df_cleaned = df.dropna(subset=features)

    if df_cleaned.empty:
        print(f"Warning: No valid data points for clustering in this DataFrame after dropping NaNs for features: {features}")
        return pd.DataFrame(), np.array([])

    X = df_cleaned[features]

    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
    df_cleaned['cluster'] = kmeans.fit_predict(X)

    print(f"\nK-Means Clustering Results for features {features} (n_clusters={n_clusters}):")
    print("Cluster Centroids:")
    print(kmeans.cluster_centers_)

    return df_cleaned, kmeans.cluster_centers_

def perform_nonlinear_regression(df, feature, target, model_func, p0=None, segments=None):
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

    plt.figure(figsize=(10, 7)) # Only one figure for all regression plots
    plt.scatter(x_data_full, y_data_full, alpha=0.5, label="Data")

    if segments:
        segment_points = sorted([min(x_data_full)] + segments + [max(x_data_full)])
        colors = plt.cm.viridis(np.linspace(0, 1, len(segment_points) - 1)) # Use a colormap for segments

        for i in range(len(segment_points) - 1):
            start = segment_points[i]
            end = segment_points[i+1]

            # Filter data for the current segment
            segment_df = df_filtered[(df_filtered[feature] >= start) & (df_filtered[feature] <= end)]
            x_segment = segment_df[feature].values
            y_segment = segment_df[target].values

            # Ensure enough data points for fitting
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
        # Single fit for the entire dataset
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
    plt.title(f"Nonlinear Regression: {target} vs {feature} (Power Function Fit)")
    plt.legend()
    plt.grid(True) # Added grid for better readability
    plt.show() # Only one plt.show() for the entire figure

# Define power function model
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
    # Add a small epsilon to x to avoid log(0) if x can be zero in power function.
    # For very small x values, np.power(0, b) can be 0 or 1 depending on b.
    # If your data naturally has 0 dollar_value, you might need to reconsider
    # the power law model or add a small offset.
    return a * np.power(x + 1e-9, b) # Added a small constant to prevent issues with x=0

if __name__ == "__main__":
    all_exports = load_data()
    labels = ['Cereals', 'Inorganic', 'Mineral', 'Ores', 'Wood']

    print("--- Head of each DataFrame ---")
    for label, df in zip(labels, all_exports):
        print(f"\n{label} Data Head:")
        print(df.head())

    print("\n--- Size of each DataFrame ---")
    for label, df in zip(labels, all_exports):
        print(f"{label} Size: {df.size}")

    print("\n--- Plotting and HDI Correlation ---")
    histogram(all_exports[0]);
    plot_scatter(all_exports[0], 'Cereals')
    plot_scatter(all_exports[1], 'Inorganic')
    plot_scatter(all_exports[2], 'Minerals')
    plot_scatter(all_exports[3], 'Ores')
    plot_scatter(all_exports[4], 'Wood')

    print("\nHDI Correlation for each resource:")
    analyze_hdi_correlation(all_exports, labels)

    print("\n--- Performing K-Means Clustering ---")
    mineral_df_clustered, mineral_centroids = perform_kmeans_clustering(
        all_exports[2],
        ['dollar_value', 'HDI_value'],
        n_clusters=3
    )

    if not mineral_df_clustered.empty:
        print("\nMineral DataFrame with Cluster Assignments (first 5 rows):")
        print(mineral_df_clustered.head())

    print("\n--- Performing Non-linear Regression (Power Law) ---")

    perform_nonlinear_regression(
        all_exports[0],
        'dollar_value',
        'HDI_value',
        model_func=power_law,
        p0=[0.1, 0.1]
    )

    perform_nonlinear_regression(
        all_exports[1],
        'dollar_value',
        'HDI_value',
        model_func=power_law,
        p0=[0.1, 0.1]
    )

    perform_nonlinear_regression(
        all_exports[2],
        'dollar_value',
        'HDI_value',
        model_func=power_law,
        p0=[0.1, 0.1]
    )

    perform_nonlinear_regression(
        all_exports[3],
        'dollar_value',
        'HDI_value',
        model_func=power_law,
        p0=[0.1, 0.1]
    )

    perform_nonlinear_regression(
        all_exports[4],
        'dollar_value',
        'HDI_value',
        model_func=power_law,
        p0=[0.1, 0.1]
    )

    # with split functionality; currently, split functionality is x-income based
    # perform_nonlinear_regression(
    #     all_exports[1],
    #     'dollar_value',
    #     'HDI_value',
    #     model_func=power_law,
    #     p0=[0.1, 0.1],
    #     segments=[1e9, 5e9]
    # )