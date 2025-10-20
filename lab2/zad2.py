import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

iris = datasets.load_iris()
x = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="FLowerType")

pca = PCA(n_components = 3)
X_pca = pca.fit_transform(x)

print("Wariancje: ")
print(pca.explained_variance_ratio_)
print(f"Suma wariancji: {sum(pca.explained_variance_ratio_):.4f}")

var_sum = np.cumsum(pca.explained_variance_ratio_)
print("Skumulowana wariancja: ", var_sum)
n_col_95 = np.argmax(var_sum >= 0.95) + 1
print(f"Liczba kolumn z zachowaniem min 95% wariancji: {n_col_95}")

plt.figure(figsize=(7, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c = y, cmap='viridis')
plt.title("PCA")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.colorbar(label = 'Typ kwiatu')
plt.show()