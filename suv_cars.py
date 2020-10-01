#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 12:49:09 2020

@author: numberphile
"""


import pandas as pd
import numpy as nm
import matplotlib.pyplot as mtp
import seaborn as sns

cars=pd.read_csv('/home/numberphile/machine learning/suv/suv_data.csv')

#how the variables relate
sns.catplot(x="Gender", y="Purchased", hue="Gender", kind="bar", data=cars)
sns.catplot(x='Purchased',y='Age', hue='Gender',kind='box', data=cars)
sns.catplot(x='Purchased',y='EstimatedSalary', hue='Gender',kind='box', data=cars)

#clean my data
null=pd.isnull(cars).sum()
 #1. drop the id column
car=cars.drop(['User ID'],axis=1)
# 2.converting gender into dummies then concertinate
gender=pd.get_dummies(car['Gender'])

result = pd.concat([car,gender], axis=1, join='inner')
car=result.drop(['Gender', 'Female'], axis=1)

#separate into features and the target variable
y=car['Purchased'].values
x=car.drop(['Purchased'], axis=1)

from sklearn.model_selection import train_test_split
x_train, x_test,y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=0)

#feature Scaling  
from sklearn.preprocessing import StandardScaler    
st_x= StandardScaler()    
x_train= st_x.fit_transform(x_train)    
x_test= st_x.transform(x_test) 

#Fitting Logistic Regression to the training set  
from sklearn.linear_model import LogisticRegression  
classifier= LogisticRegression(random_state=0)  
classifier.fit(x_train, y_train) 

#Predicting the test set result  
y_pred= classifier.predict(x_test)  


#Creating the Confusion matrix  
from sklearn.metrics import confusion_matrix  
cm=confusion_matrix(y_test, y_pred)



# To compute the accuracy of our model
from sklearn.metrics import accuracy_score
f1=accuracy_score(y_test, y_pred)


