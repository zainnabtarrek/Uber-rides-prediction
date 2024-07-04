import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

data = pd.read_csv("uber_dataset.csv")
print(data.head())

X = data.drop(["Numberofweeklyriders"], axis=1)
y = data["Numberofweeklyriders"]

model = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
model.fit(X_train, y_train)

pickle.dump(model,open('model.pk','wb'))
my_model=pickle.load(open('model.pk','rb'))

y_pred = my_model.predict(X_test)
print(y_pred)