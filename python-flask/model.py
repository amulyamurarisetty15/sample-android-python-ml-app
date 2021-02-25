import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold


data = pd.read_csv('Charitydonor.csv')

y = data['income']
X = data.drop('income', axis = 1)

kf = KFold(n_splits=10)
kf.get_n_splits(X)
for train_index, test_index in kf.split(X,y):
    x_train, x_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]



model=GradientBoostingClassifier()
model.fit(x_train,y_train)

pickle.dump(model, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[1,2,3,4,5,6,7,8,9,10,11,12,13]]))
