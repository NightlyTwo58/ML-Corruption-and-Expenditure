import os
import sys
import pandas as pd

def read_csv_file(filename):
    """Reads a CSV file into a pandas DataFrame."""
    try:
        df = pd.read_csv(filename)
        return df
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.", file=sys.stderr)
        return None

def filter_hdi_ratio(df, min_hdi=0.1, max_hdi=1):
    """Filters DataFrame rows based on the dollar_per_capita / HDI_value ratio and optional max HDI."""
    df = df.copy()
    df['ratio'] = df['dollar_per_capita'] / df['HDI_value']
    filtered_df = df[df['ratio'] > min_hdi]
    if max_hdi is not None:
        filtered_df = filtered_df[filtered_df['HDI_value'] <= max_hdi]
    return filtered_df

def filter_by_cluster(df, clusters):
    """Filters DataFrame rows to only include specified clusters."""
    df = df.copy()
    filtered_df = df[df['cluster'].isin(clusters)]
    return filtered_df


def save_filtered_df(df, filename, suffix, save_path=None):
    """
    Saves filtered DataFrame to a CSV file and prints summary of countries and years.

    Args:
        df (pd.DataFrame): The filtered DataFrame to save.
        filename (str): Original filename (used to construct default save name if save_path is None).
        suffix (str): Suffix to append to the filename.
        save_path (str, optional): Full path to save the CSV. If None, default naming is used.
    """
    if save_path is None:
        base, ext = os.path.splitext(filename)
        newfile = f"{base}_{suffix}{ext}"
    else:
        newfile = save_path

    print("Countries (years) in filtered data:")
    grouped = df.groupby('country_code_letter')['year'].apply(list)
    for country, years in grouped.items():
        years_str = ', '.join(map(str, years))
        print(f"Country: {country} ({years_str})")

    df.to_csv(newfile, index=False, float_format="%.6f")
    print(f"Filtered data saved to '{newfile}'")

if __name__ == "__main__":
    filename = 'data/clustering_results/mineral.csv'

    # Read file
    df = read_csv_file(filename)

    # Filter by cluster
    filtered_cluster_df = filter_by_cluster(df, clusters=[0])
    # Filter by HDI ratio
    filtered_hdi_df = filter_hdi_ratio(filtered_cluster_df, min_hdi=0.1, max_hdi=1)
    save_filtered_df(filtered_hdi_df, filename, "", "data/clustering_results_2/mineral_2_group_0_filtered_0.1.csv")
