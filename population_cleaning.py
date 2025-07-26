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
    population = pd.read_csv("data/population_data.csv")
    return [cereals, inorganic, mineral, ores, wood], population, ['Cereals', 'Inorganic', 'Mineral', 'Ores', 'Wood']

def add_population_data(export_df, population_df, output_filename="output_with_population.csv"):
    """
    Adds population data to a single natural resource export DataFrame and saves it to a CSV.

    Args:
        export_df (pandas.DataFrame): A single DataFrame representing
                                      natural resource exports.
        population_df (pandas.DataFrame): The population DataFrame.
        output_filename (str): The name of the CSV file to save the result.

    Returns:
        None
    """
    # Melt the population DataFrame to long format for easier merging
    # Identify year columns (assuming they are numeric and not 'Country Code')
    id_vars = ['Country Code']
    value_vars = [col for col in population_df.columns if col not in id_vars]

    # Convert year columns to numeric, handling potential errors
    for col in value_vars:
        population_df[col] = pd.to_numeric(population_df[col], errors='coerce')

    # Melt the DataFrame
    population_long = population_df.melt(
        id_vars=['Country Code'],
        value_vars=value_vars,
        var_name='year',
        value_name='population'
    )
    population_long['population'] = population_long['population'].round()

    # Convert 'year' column in population_long to integer type
    population_long['year'] = pd.to_numeric(population_long['year'], errors='coerce').astype('Int64')
    # Rename 'Country Code' to 'country_code_letter' for consistent merging
    population_long.rename(columns={'Country Code': 'country_code_letter'}, inplace=True)
    # Ensure 'year' column in export data is also integer type
    export_df['year'] = pd.to_numeric(export_df['year'], errors='coerce').astype('Int64')

    merged_df = pd.merge(
        export_df,
        population_long,
        on=['year', 'country_code_letter'],
        how='left'
    )

    merged_df['population'] = merged_df['population'].round()
    merged_df['population'] = merged_df['population'].astype('Int64')
    print(merged_df['population'].dtype)

    try:
        merged_df.to_csv(output_filename, index=False)
        print(f"Successfully saved data with population to {output_filename}")
    except Exception as e:
        print(f"Error saving file {output_filename}: {e}")

if __name__ == "__main__":
    print("--- Head of each DataFrame ---")
    exports, population, labels = load_data()
    for resource, label in zip(exports, labels):
        add_population_data(resource, population, "data/Exports Pop Comb/" + label + "_population.csv")