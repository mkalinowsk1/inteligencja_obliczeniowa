import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# a) Podziel w losowy sposób bazę danych irysów na zbiór treningowy i zbiór testowy w proporcjach 70%/30%.
# Wyświetl oba zbiory. Podziel te zbiory na cztery części (inputy i class), jeśli jest taka potrzeba.


df = pd.read_csv("iris.csv")
all_inputs = df[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']].values
all_classes = df['variety'].values
(train_inputs, test_inputs, train_classes, test_classes) = train_test_split(
    all_inputs, all_classes, train_size=0.7, random_state=288549)


# print(train_inputs)
# print(train_classes)

# b) Zainicjuj drzewo decyzyjne metodą DecisionTreeClassifier.

clf = tree.DecisionTreeClassifier()

# c) Wytrenuj drzewo decyzyjne na zbiorze treningowym, wykorzystując funkcję fit.

clf = clf.fit(train_inputs, train_classes)

# d) Wyświetl drzewo w formie tekstowej i/lub w formie graficznej.

tree.plot_tree(clf)
plt.show()

# e) Dokonaj ewaluacji klasyfikatora: sprawdź jak drzewo poradzi sobie z rekordami ze zbioru testowego. Wyświetl
# dokładność klasyfikatora, czyli procent poprawnych odpowiedzi. Wykorzystaj do tego funkcję score lub predict.

print(f"{clf.score(test_inputs, test_classes):.2%}")

# f) Wyświetl macierz błędów (ang. confusion matrix), która zestawia liczby błędnych i poprawnych odpowiedzi, dla
# wszystkich klas

predictions = clf.predict(test_inputs)
cm = confusion_matrix(test_classes, predictions, labels=clf.classes_)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Macierz błędów dla drzewa decyzyjnego")
plt.show()

