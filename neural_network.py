# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 16:12:08 2020

@author: swati bansal
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Churn_Modelling.csv')
df.head()

X = df.iloc[:,3:13] # Independent Variable
y = df.iloc[:,13]   #Dependent Variable

df.shape

#Creating the dummies varaibles
geography = pd.get_dummies(X["Geography"],drop_first=True)
gender = pd.get_dummies(X["Gender"],drop_first=True)

#Concating the dataframe
X = pd.concat([X,geography,gender],axis=1)

#Drip the Unnecessary Columns
X = X.drop(['Geography','Gender'],axis=1)

#Spliiting the dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part-2 : Now lets make the ANN model

#Importing the keras libraries and packages

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU, ELU
from keras.layers import Dropout

#Inializing the ANN:
classifier = Sequential()

#Adding the input layer and the first layer
classifier.add(Dense(units =6, kernel_initializer='he_uniform', activation='relu',input_dim=11))

#Adding the second hidden layer
classifier.add(Dense(units=6,kernel_initializer='he_uniform',activation='relu'))

#Adding the output layer
classifier.add(Dense(units=1,kernel_initializer='glorot_uniform',activation='sigmoid'))

#Compiling the ANN:
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#Fitting the ANN into the training set
model_history = classifier.fit(X_train,y_train,validation_split =0.33,batch_size=10,nb_epoch=100)


#Part-3 Making the predictions and evaluting the model
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

#Calculating the accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_pred,y_test)








