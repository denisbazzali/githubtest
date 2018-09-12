import numpy as np
import seaborn as sns #per fare grafici
import matplotlib.pyplot as plt #grafici
import pandas as pd
from random import randint
#import del dataset
dataset = pd.read_csv('wineset.csv') #doppia slash, ricordarsi l'estensione del file


from sklearn.metrics import confusion_matrix #htsest
from sklearn.metrics import classification_report

## X che esce da qui è un Array di vettori che contengono ognuno i 13 attributi di ogni riga (tutti numeri)
X = dataset.iloc[:, :-1].values #subset delle variabili indipendenti, tutte le righe e tutte le colonne - l'ultima (la specie)

## y che esce da qui è un Array di stringhe che contengono le classi di ogni riga
y = dataset.iloc[:, -1].values #subset delle variabili indipendenti

## Training e test vengono generati
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.20, random_state=randint(0, 10))


# procedura di normalizzazione
x_train_norm = np.zeros(X_train.shape)
x_test_norm = np.zeros(X_test.shape)
for i in range(X_train.shape[1]): # Loop over all columns
    mu = np.mean(X[:, i]) # Compute the mean
    sigma = np.std(X[:, i]) # Compute the standard deviation
    # Normalize a column in the training set
    x_train_norm[:, i] = (X_train[:, i] - mu) / sigma
    # Normalize the test set (using mu and sigma from the training set)
    x_test_norm[:, i] = (x_test_norm[:, i] - mu) / sigma


import keras

from keras.layers import Dense
from keras.models import Sequential

model = Sequential()

model.add(Dense(13,activation='softmax', input_dim=13))
model.add(Dense(3,activation='softmax', input_dim=13))
#model.add(Dense(3, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

from sklearn import preprocessing

encoder = preprocessing.LabelEncoder()
t_y_train = encoder.fit_transform(y_train)
t_y_test = encoder.fit_transform(y_test)


y_train_cat = keras.utils.to_categorical(t_y_train)
y_test_cat = keras.utils.to_categorical(t_y_test)

model.fit(x_train_norm, t_y_train,
          epochs=50)

score = model.evaluate(x_test_norm, t_y_test, batch_size=x_test_norm.shape[0])
score

predicted = model.predict(x_test_norm)


from sklearn.metrics import confusion_matrix

print(confusion_matrix(t_y_test,predicted))




from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train,y_train)
print(classifier)
y_pred = classifier.predict(X_test)
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
from sklearn.metrics import accuracy_score
print('accuracy is: '+ str(accuracy_score(y_pred,y_test)))
