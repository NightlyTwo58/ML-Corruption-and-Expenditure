import numpy as np
import pandas as pd
from sklearn.metrics import recall_score
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.svm import SVC
# import seaborn as sns

# TODO 1: fill NaN values
# Specify the file path
file_path = r"C:\Users\xuena\Downloads\HDI_data_global_trimmed.csv"
df = pd.read_csv(file_path)
df.fillna("NaN")
print(df)


# TODO 2: flip HDI dataset row to column
# TODO 3: merge datasets