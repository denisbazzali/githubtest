import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(color_codes=True)
import pandas as pd

dataset = pd.read_csv('Iris.csv')



## Degli esempi di manipolazione dei dati #####
print(dataset.head())
ds2 = dataset[dataset.SepalLengthCm < 5]
print(ds2)
print(ds2.size)
ds3 = ds2.groupby(by=['Species']).mean()
print(ds3)
ds3 = ds3.drop('Id', 1)
ds3.plot(kind='box')
######################################### #####

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

x = dataset.iloc[:, :-1].values #prendere tutti i valori tranne l'ultima colonna
y = dataset.iloc[:, -1].values #prendere solo l'ultima colonna
