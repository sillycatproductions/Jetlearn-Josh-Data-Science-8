import pandas as pd #helps read csv files
import numpy as np #numerical python - good with numerical data
import matplotlib.pyplot as plt #plots graphs based on data
from sklearn.tree import DecisionTreeClassifier #creates machine learning algorithm
from sklearn.model_selection import train_test_split #helps splitting data to test and train
from sklearn import metrics 

dataf = pd.read_csv('data.csv') #reads the file
print(dataf.head()) #prints the first 5 columns of the file

print(dataf.info()) #prints info about the file (e.g. type of values, number of items, null or non-null)

dataf['species'] = dataf['species'].replace({'setosa':0, 'versicolor':1, 'virginica':2}) #replaces obj (str) with float / int
print(dataf.head(10))

plt.subplot(221)
plt.scatter(dataf['petal_length'], dataf['species'], s = 10, c = 'green', marker = 'o')

plt.subplot(222)
plt.scatter(dataf['sepal_length'], dataf['species'], s = 10, c = 'blue', marker = 'o')

plt.subplot(223)
plt.scatter(dataf['petal_width'], dataf['species'], s = 10, c = 'red', marker = 'o')

plt.subplot(224)
plt.scatter(dataf['sepal_width'], dataf['species'], s = 10, c = 'orange', marker = 'o')
plt.show()

y = dataf['species']
x = dataf.drop('species', axis = 1)

print(x.head())
print(y.head())

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 1)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

model = DecisionTreeClassifier(max_depth = 3, random_state = 1)
model.fit(x_train, y_train)

pred = model.predict(x_test)
print('Accuracy:', metrics.accuracy_score(pred, y_test))
