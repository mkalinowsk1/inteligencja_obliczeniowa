import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv("iris_big.csv")
(train_set, test_set) = train_test_split(df.values, train_size=0.7, random_state=288549)


def classify_iris(sl, sw, pl, pw):
    if pw <= 0.6:
        return("setosa")
    elif pw > 1.6:
        return("virginica")
    else:
        return("versicolor")
    

good_predictions = 0
len = test_set.shape[0]

for i in range(len):
    if classify_iris(test_set[i][0], test_set[i][1], test_set[i][2], test_set[i][3]) == test_set[i][4]:
        good_predictions = good_predictions + 1
print(good_predictions)
print(good_predictions/len*100, "%")

#print(train_set)
