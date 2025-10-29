import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# a) Podziel w losowy sposób bazę danych na zbiór treningowy (70%) i testowy (30%).

df = pd.read_csv("iris_big.csv")

X = df[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']].values
y = df['variety'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7, random_state=288549)

def evaluate_model(model, model_name):
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(model_name)
    print(f"Dokładność: {accuracy:.2%}")

    predictions = model.predict(X_test)
    cm = confusion_matrix(y_test, predictions, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Macierz błędów - {model_name}")
    plt.show()

# Uruchom każdy z klasyfikatorów wykorzystując paczki i dokonaj ewaluacji ma zbiorze testowym wyświetlając
# procentową dokładność i macierz błędu.

for k in [3, 5, 11]:
    knn = KNeighborsClassifier(n_neighbors=k)
    evaluate_model(knn, f"k-NN (k={k})")

nb = GaussianNB()
evaluate_model(nb, "Naive Bayes (GaussianNB)")
