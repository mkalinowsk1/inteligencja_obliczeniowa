import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

iris = datasets.load_iris()
x = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="FLowerType")

X_orig = x[['sepal length (cm)', 'sepal width (cm)']]

scaler_minmax = MinMaxScaler()
X_minmax = scaler_minmax.fit_transform(X_orig)

scaler_zscore = StandardScaler()
X_zscore = scaler_zscore.fit_transform(X_orig)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].scatter(X_orig['sepal length (cm)'], X_orig['sepal width (cm)'], c=y, cmap='viridis')
axes[0].set_title("Dane oryginalne")

axes[1].scatter(X_minmax[:, 0], X_minmax[:, 1], c=y, cmap='viridis')
axes[1].set_title("Normalizacja Min-Max")

axes[2].scatter(X_zscore[:, 0], X_zscore[:, 1], c=y, cmap='viridis')
axes[2].set_title("Standaryzacja Z-score")

for ax in axes:
    ax.set_xlabel("Sepal length")
    ax.set_ylabel("Sepal width")

plt.tight_layout()
plt.show()

print("\nStatystyki oryginalne:")
print(X_orig.describe())

print("\nStatystyki po Min-Max:")
print(pd.DataFrame(X_minmax, columns=X_orig.columns).describe())

print("\nStatystyki po Z-score:")
print(pd.DataFrame(X_zscore, columns=X_orig.columns).describe())