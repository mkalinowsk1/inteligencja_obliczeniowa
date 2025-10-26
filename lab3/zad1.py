import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv("iris.csv")
(train_set, test_set) = train_test_split(df.values, train_size=0.7, random_state=288549)


def classify_iris(sl, sw, pl, pw):
    if pw <= 0.6:
        return("Setosa")
    elif pw > 1.6:
        return("Virginica")
    else:
        return("Versicolor")
    

good_predictions = 0
len = test_set.shape[0]
print(classify_iris(train_set[2][0], train_set[2][1], train_set[2][2], train_set[2][3]))

for i in range(len):
    sl = float(train_set[i][0])
    sw = float(train_set[i][1])
    pl = float(train_set[i][2])
    pw = float(train_set[i][3])
    if classify_iris(sl, sw, pl , pw) == test_set[i][4]:
        good_predictions = good_predictions + 1
print(good_predictions)
print(good_predictions/len*100, "%")

#print(train_set)
