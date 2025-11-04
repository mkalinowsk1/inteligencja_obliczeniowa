import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv('diabetes.csv')
print("Informacje o zbiorze danych:")
print(df.info())
print("\\nPierwsze wiersze:")
print(df.head())
print("\\nStatystyki:")
print(df.describe())

print("\\nBraki danych:")
print(df.isnull().sum())

columns_to_check = ['glucose-concentr', 'blood-pressure', 'skin-thickness', 'insulin', 'mass-index']
for col in columns_to_check:
    if col in df.columns:
        zeros = (df[col] == 0).sum()
        if zeros > 0:
            print(f"{col}: {zeros} wartości zerowych (prawdopodobnie błędne)")
            df[col] = df[col].replace(0, df[col].median())

# b) Sprawdzenie danych kategorialnych
print("\\nTypy danych:")
print(df.dtypes)
categorical_cols = df.select_dtypes(include=['object']).columns
if len(categorical_cols) > 0:
    print(f"Kolumny kategorialne: {list(categorical_cols)}")
    # Konwersja na numeryczne
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

# c) Normalizacja
X = df.drop('class', axis=1) 
y = df['class']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# d) Podział na zbiory
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# e) Testowanie klasyfikatorów
classifiers = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'kNN (k=5)': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB(),
    'Neural Net (5,)': MLPClassifier(hidden_layer_sizes=(5,), max_iter=1000, random_state=42),
    'Neural Net (10, 5)': MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=1000, random_state=42),
    'Neural Net (15, 10, 5)': MLPClassifier(hidden_layer_sizes=(15, 10, 5), max_iter=1000, random_state=42),
    'Neural Net tanh': MLPClassifier(hidden_layer_sizes=(10, 5), activation='tanh', max_iter=1000, random_state=42),
}

results = {}

for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    results[name] = {
        'accuracy': accuracy,
        'confusion_matrix': cm
    }
    
    print(f"\\n{name}:")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Macierz błędów:\\n{cm}")

# f) Wykres słupkowy
plt.figure(figsize=(12, 6))
names = list(results.keys())
accuracies = [results[name]['accuracy'] * 100 for name in names]

plt.bar(names, accuracies)
plt.xlabel('Klasyfikator')
plt.ylabel('Dokładność (%)')
plt.title('Porównanie dokładności klasyfikatorów - Diabetes Dataset')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 100)
plt.tight_layout()
plt.show()

# g) Analiza błędów FP i FN
print("\\n--- ANALIZA BŁĘDÓW ---")
print("FN (False Negative): Chory zaklasyfikowany jako zdrowy - POWAŻNIEJSZY błąd!")
print("FP (False Positive): Zdrowy zaklasyfikowany jako chory - mniej poważny")
print("\\nW kontekście diagnozy cukrzycy:")
print("- FN jest groźniejszy: osoba chora nie dostanie leczenia")
print("- FP jest mniej groźny: osoba zdrowa zostanie skierowana na dodatkowe badania")

for name, res in results.items():
    cm = res['confusion_matrix']
    tn, fp, fn, tp = cm.ravel()
    print(f"\\n{name}:")
    print(f"  FP (False Positive): {fp}")
    print(f"  FN (False Negative): {fn}")
    
min_fn_classifier = min(results.items(), 
                        key=lambda x: x[1]['confusion_matrix'].ravel()[2])
print(f"Klasyfikator minimalizujący FN: {min_fn_classifier[0]}")