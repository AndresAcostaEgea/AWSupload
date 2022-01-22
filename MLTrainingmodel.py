#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('train.csv')

def getTitle(name):
    if "." in name:
        return name.split(",")[1].split(".")[0].strip()
    else: 
        return "No title in name"

def shortTitles(x):
    title = x["Title"]
    if title in ['Capt','Col','Major']:
        return "Officer"
    elif title in ['the Countess', 'Don', 'Sir', 'Jonkheer', 'Lady']:
        return "Royalty"
    elif title == 'Mme':
        return "Mrs"
    elif title in ['Mlle', 'Ms']:
        return "Miss"
    else:
        return title

df["Title"] = df['Name'].map(lambda x: getTitle(x))
df["Title"] = df.apply(shortTitles, axis=1)

df.drop("Name", axis=1, inplace=True)
df.drop("Ticket", axis=1, inplace=True)
df.drop("Cabin", axis=1, inplace=True)
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna("S", inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)
df.Sex.replace(('male','female'), (0,1), inplace=True)
df.Embarked.replace(('S','C','Q'), (0,1,2), inplace=True)
df.Title.replace(('Mr','Miss','Mrs','Master','Dr','Rev','Royalty','Officer'), (0,1,2,3,4,5,6,7), inplace=True)

x = df.drop(['Survived','PassengerId'], axis=1)
y = df['Survived']
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1)

randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)

pickle.dump(randomforest, open('titanic.model.sav', 'wb'))

