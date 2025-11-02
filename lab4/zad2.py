import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import math

iris = load_iris()
x = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


print("Klasy irysów: ")
for i, name in enumerate(iris.target_names):
    print(f"  {i} = {name}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

architectures = [
    (5,),           
    (5, 3),         
    (5, 3, 2)       
]

results = []

for i, arch in enumerate(architectures, 1):
    print(f"\nArchitektura {i}: {arch}")
    
    mlp = MLPClassifier(
        hidden_layer_sizes=arch,
        max_iter=1000,
        random_state=42,
        activation='relu'
    )
    
    mlp.fit(X_train_scaled, y_train)
    y_pred = mlp.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Macierz błędów:\n{cm}")
    
    results.append({
        'architecture': str(arch),
        'accuracy': accuracy,
        'confusion_matrix': cm
    })

best_arch = max(results_task2, key=lambda x: x['accuracy'])
print(f"\n*** Najlepsza architektura: {best_arch['architecture']} "
      f"z accuracy = {best_arch['accuracy']:.4f} ***")