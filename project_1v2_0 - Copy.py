import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress
from scipy.stats import f_oneway
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

def load_data():
    budget_data = pd.read_csv("data/government_expenditures/bugetary_data.csv")
    return budget_data

if __name__ == "__main__":
    budget_data = load_data()
    labels = budget_data[0, :]