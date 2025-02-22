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
plt.show()
