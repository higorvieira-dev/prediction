import pandas as pd
import pandas as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import preprocessing
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error


data=pd.read_csv("rainfall.csv")

print("Data heads:")
print(data.head())
print("Null values in the dataset before preprocessing:")
print(data.isnull().sum())
print(" Filling null values with of that particular column")

data=data.fillna(np.mean(data))

print("Mean of data:")
print(np.mean(data))
print("Null values in the dataset before preprocessing:")
print(data.isnull().sum())
print("\n\nShape:", data.shape)
print("info:")
print(data.info())
print("Group By:")

data.groupby("DEVLOG").size()

print("Co-Variance = ", data.cov())
print("Co-Relation =", data.corr())

corr_cols=data.corr()['ANNUAL'].sort_values()[::-1]

print("Index of correlation columns:", corr_cols.index)
print("Scatter plot of annual and junuary attributes")

plt.scatter(data.ANNUAL, data.JAN)