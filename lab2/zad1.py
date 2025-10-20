import pandas as pd
import numpy as np

df = pd.read_csv("iris_with_errors.csv")

print("Błędne dane: ")
print(df.head())

# a) Policz ile jest w bazie brakujących lub nieuzupełnionych danych. Wyświetl statystyki bazy danych z błędami.

print("Liczba brakujących danych: ")
print(df.isnull().sum())

print("Statystyki bazy z błędami:")
print(df.describe(include="all"))

# b) Sprawdź czy wszystkie dane numeryczne są z zakresu (0; 15). Dane spoza zakresu muszą być poprawione. Możesz
# tutaj użyć metody: za błędne dane podstaw średnią (lub medianę) z danej kolumny. 

num_cols = df.select_dtypes(include=[np.number]).columns

for col in num_cols:
	mean_val = df[col].mean()
	df[col] = df[col].apply(lambda x: mean_val if (x < 0 or x > 15 or pd.isna(x)) else x)

# c) Sprawdź czy wszystkie gatunki są napisami: „Setosa”, „Versicolor” lub „Virginica”. Jeśli nie, wskaż jakie popełniono
# błędy i popraw je własną (sensowną) metodą.

variety_col = df.select_dtypes(exclude=[np.number]).columns[2]

print(f"\nNazwy gatunków przed poprawą:\n{df[variety_col].unique()}")

names = {
    'setosa': 'Setosa',
    'Versicolour': 'Versicolor',
    'virginica': 'Virginica'
}

df[variety_col] = df[variety_col].replace(names)

print(f"\nNazwy gatunków po poprawie:\n{df[variety_col].unique()}")