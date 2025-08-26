# uncomment these if you want to rerun the clustering data beforehand
# import mineral_clusreg
# import cereals_clusreg
# import ore_clusreg
# import wood_clusreg
# import inorganic_clusreg

import os
import project_1v2_0
import numpy as np
import sys
import pandas as pd

def filter_hdi_ratio(filename, threshold, max_hdi=None):
    """
    Filters a CSV file, keeping rows where the dollar_per_capita / HDI_value
    ratio is greater than a specified threshold. An optional maximum HDI value
    can also be provided to further filter the data.

    Args:
        filename (str): The path to the input CSV file.
        threshold (float): The minimum ratio for rows to be kept.
        max_hdi (float, optional): An optional maximum HDI value. Rows with
                                   an HDI value greater than this will be removed.
                                   Defaults to None.
    """
    try:
        df = pd.read_csv(filename)

        df.loc[:, 'ratio'] = df['dollar_per_capita'] / df['HDI_value']
        filtered_df = df[df['ratio'] > threshold].copy()
        if max_hdi is not None:
            filtered_df = filtered_df[filtered_df['HDI_value'] <= max_hdi]

        print("Countries (years) that meet threshold and HDI max")
        unique_countries_years = filtered_df.groupby('country_code_letter')['year'].apply(list)
        for country, years in unique_countries_years.items():
            years_str = ', '.join(map(str, years))
            print(f"Country: {country} ({years_str})")

        name, ext = os.path.splitext(filename)
        new_filename = f"{name}_filtered_{threshold}{ext}"

        filtered_df.to_csv(new_filename, index=False, float_format="%.6f")
        print(f"File '{new_filename}' successfully filtered and written.")

    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.", file=sys.stderr)

def filter_by_cluster(clusters, filename):
    """
    Removes all clusters other than those specified in filename, prints to a new .csv file if filename is specified.

    Args:
        filename (str): Path to the input CSV file.
        threshold (float): The minimum ratio for rows to be kept.
    """
    try:
        df = pd.read_csv(filename)
        country_codes_df = pd.read_csv('data/country_codes.csv')

        filtered_df = df[df['cluster'].isin(clusters)].copy()
        base, ext = os.path.splitext(filename)
        newfile = f"{base}_group_{'_'.join(str(n) for n in clusters)}{ext}"

        print("Countries (years) filtered")
        unique_countries_years = filtered_df.groupby('country_code_letter')['year'].apply(list)
        for country, years in unique_countries_years.items():
            years_str = ', '.join(map(str, years))
            print(f"Country: {country} ({years_str})")
        filtered_df.to_csv(newfile, index=False)
        print(f"Filtered data saved to {newfile}")
    except FileNotFoundError:
        print(f"Error: File was not found.", file=sys.stderr)

# Example usage:
if __name__ == "__main__":
    filter_by_cluster([0], 'data/clustering_results/mineral.csv')
    filter_hdi_ratio('data/clustering_results/mineral_group_0.csv', 0.1, 1)
    filter_by_cluster([1], 'data/clustering_results/mineral.csv')