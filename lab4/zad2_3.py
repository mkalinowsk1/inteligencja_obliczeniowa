import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv("iris_big.csv")
all_inputs = df[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']].values
all_classes = df['variety'].values
(X_train, X_test, y_train, y_test) = train_test_split(
    all_inputs, all_classes, train_size=0.7, random_state=288549)



scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

architectures = [
    (5,),           
    (5, 3),         
    (5, 3, 2)       
]

results2 = []

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
    
    results2.append({
        'architecture': str(arch),
        'accuracy': accuracy,
        'confusion_matrix': cm
    })

best_arch = max(results2, key=lambda x: x['accuracy'])
print(f"Najlepsza architektura: {best_arch['architecture']} "
      f"z accuracy = {best_arch['accuracy']:.4f} ")


print("==========================================================================")
print("zad3")
print("==========================================================================")

y_train_2d = y_train.reshape(-1, 1)
y_test_2d = y_test.reshape(-1, 1)

encoder = OneHotEncoder(sparse_output=False)
y_train_encoded = encoder.fit_transform(y_train_2d)
y_test_encoded = encoder.transform(y_test_2d)

print(f"Oryginalna etykieta: {y_train[0]} ({all_classes[0]})")
print(f"Po one-hot encoding: {y_train_encoded[0]}")

architectures_onehot = [
    (5,),           
    (5, 3),         
    (5, 3, 2)       
]

results3 = []

for i, arch in enumerate(architectures_onehot, 1):
    print(f"\nArchitektura {i}: {arch} -> output: 3 neurony")
    
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
    
    results3.append({
        'architecture': str(arch),
        'accuracy': accuracy,
        'confusion_matrix': cm
    })

print("\n--- PORÓWNANIE: Zadanie 2 vs Zadanie 3 ---")
for i in range(len(results2)):
    print(f"Architektura {architectures[i]}:")
    print(f"  Zadanie 2 (zwykłe): {results2[i]['accuracy']:.4f}")
    print(f"  Zadanie 3 (one-hot): {results3[i]['accuracy']:.4f}")