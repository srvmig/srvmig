# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 14:35:06 2021

@author: ASUS
"""

#%%
# Import all packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



#%%
# Load datas
train = pd.read_csv('D:/Datasets/titanic/train.csv')
train.head(10)

test = pd.read_csv('D:/Datasets/titanic/test.csv')
test.head(10)

#%%
# Checking datasets
train.info()
test.info()

train.isnull().sum()
train.isnull().mean()

test.isnull().sum()
test.isnull().mean()

train.nunique()
test.nunique()

#%%
# Explanatory Analysis
## Sex
train['Sex'].value_counts()

#%%%
train.groupby('Sex')['Survived'].mean()

#%%%
pd.crosstab(train['Sex'], train['Survived'])

#%%%
fig, axis = plt.subplots(nrows = 1, ncols = 2, figsize = (10,4))
women = train[train.Sex == 'female']
men = train[train.Sex == 'male']
w_alive = women[women['Survived'] == 1]['Age'].dropna()
w_died = women[women['Survived']== 0]['Age'].dropna()

ax = sns.histplot(w_alive, bins = 18, label='Survived', ax = axis[0],
                  element = 'step', alpha = 0.5, kde= False)
ax = sns.histplot(w_died, bins = 40, label = 'Not survived', ax = axis[0], 
                  element = 'step', color = 'orange', alpha = 0.5, kde= False)
ax.legend()
ax.set_title('Female')

m_alive = men[men['Survived'] == 1].Age.dropna()
m_died = men[men['Survived'] == 0].Age.dropna()
ax = sns.histplot(m_alive, bins = 18, label = 'Survived', ax = axis[1], 
                  element = 'step', alpha = 0.5, kde= False)
ax = sns.histplot(m_died, bins = 40, label = 'Not survived', ax = axis[1], 
                  element = 'step', color = 'orange', alpha = 0.5, kde= False)
ax.legend()
ax.set_title('Male')
plt.show()
plt.close()

#%%%
FacetGrid = sns.FacetGrid(train, row='Embarked', height = 5, aspect =2)
FacetGrid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette = None,
              order = None, hue_order = None)
FacetGrid.add_legend()
plt.show()
plt.close()

#%%%
sns.barplot(x= 'Pclass', y= 'Survived', data = train)
plt.show()
plt.close()

grid = sns.FacetGrid(train, col= 'Survived', row= 'Pclass', height= 2,
                     aspect = 1.6)
grid.map(plt.hist, 'Age', alpha = 0.5, bins = 20)
grid.add_legend()
plt.show()
plt.close()

#%%%
grid = sns.FacetGrid(train, col= 'Survived', row= 'Pclass', height= 2,
                     hue = 'Sex', aspect = 1.6)
grid.map(plt.hist, 'Age', alpha = 0.5, bins = 20)
grid.add_legend()
plt.show()
plt.close()

#%%%
## Pclass
train['Pclass'].value_counts()
plt.bar(x = train['Pclass'], height= train['Fare'])
plt.show()
plt.close()

#%%%
train.groupby('Pclass')['Survived'].mean()

#%%%
pd.crosstab(train['Pclass'], train['Survived'])

#%%%
## Age
train['Age'].isnull().sum()

## Age has 177 data missing. Try to fill the gap


#%%%
copy_train = train.copy()
copy_test = test.copy()

#%%%
title = []
for i in train['Name']:
    tl = i.split(', ')[1].split('. ')[0]
    title.append(tl)

copy_train['Title'] = title

title = []
for i in test['Name']:
    tl = i.split(', ')[1].split('. ')[0]
    title.append(tl)

copy_test['Title'] = title

#%%
## Family member
copy_train['Family'] = copy_train['SibSp'] + copy_train['Parch']
copy_test['Family'] = copy_test['SibSp'] + copy_test['Parch']

#%%%
## Cheking child survival
child = copy_train[copy_train['Age'].between(0, 5)][['Age', 'Title', 'Sex',
                                                      'Survived', 'Pclass']]

#%%%
## Visualize child age group by sex
fig, ax = plt.subplots(figsize = (10,6))
ax = sns.countplot(x = 'Age',  hue = 'Sex', data = child)
plt.show()
plt.close()

#%%%
## Visualize child Survived group by sex
fig, ax = plt.subplots(figsize = (10,6))
ax = sns.countplot(x = 'Survived', hue = 'Sex', data = child)
plt.show()
plt.close()

#%%%
## Visualize child Survived group by class
fig, ax = plt.subplots(figsize = (10,6))
ax = sns.countplot(x = 'Survived', hue = 'Pclass', data = child)
plt.show()
plt.close()

#%%%
## Child survival in 1st & class is significant.
## So, we can convert sex of 0-5 group as child
copy_train.loc[copy_train['Age'] <= 12, 'Sex'] = 'child'
copy_test.loc[copy_test['Age'] <= 12, 'Sex'] = 'child'

#%%%
## Check titles
copy_train['Title'].value_counts()
copy_train[copy_train['Title'] == 'Dr']
copy_train[copy_train['Title'] == 'Rev']
copy_train[copy_train['Title'] == 'Major']
copy_train[copy_train['Title'] == 'Col']
copy_train[copy_train['Title'] == 'Capt']

# OR

copy_train[copy_train['Title'].isin(['Dr', 'Rev', 'Major', 'Col', 'Capt'])]

#%%%
fig, ax = plt.subplots(figsize = (10,6))
ax = sns.countplot(x = 'Survived', hue = 'Title',
                   data = copy_train[copy_train['Title'].isin(['Dr', 'Rev',
                                                               'Major', 'Col',
                                                               'Capt'])])
plt.show()
plt.close()

#%%%
## Searching by special title is not significant. 
## So we can it as it is
## Change other titles
copy_train['Title'].isnull().sum()
copy_train.groupby('Title')['Survived'].sum()

copy_train['Title'] = copy_train['Title'].replace(['Ms','Mlle', 'Mme'], 'Miss') 
copy_train['Title'] = copy_train['Title'].replace(['Lady','the Countess', 'Dona'],
                                                  'Mrs')
copy_train['Title'] = copy_train['Title'].replace(['Jonkheer', 'Don', 'Sir'],
                                                  'Mr')

#%%%
copy_train.groupby('Title')['Survived'].sum()
copy_train.groupby('Title')['Age'].mean()

#%%%
copy_train['Age'].fillna(copy_train['Title'].map(copy_train.groupby('Title')['Age'].mean()), inplace= True)
copy_train.info()

#%%%
copy_test['Title'].isnull().sum()

#%%%
copy_test['Title'] = copy_test['Title'].replace(['Ms','Mlle', 'Mme'], 'Miss') 
copy_test['Title'] = copy_test['Title'].replace(['Lady','the Countess', 'Dona'],
                                                  'Mrs')
copy_test['Title'] = copy_test['Title'].replace(['Jonkheer', 'Don', 'Sir'],
                                                  'Mr')

#%%%
copy_test.groupby('Title')['Age'].mean()

#%%%
copy_test['Age'].fillna(copy_test['Title'].map(copy_test.groupby('Title')['Age'].mean()), inplace= True)

#%%%
## Check ticket and embarked
copy_train['Embarked'].unique()

#%%%
copy_train[(copy_train['Embarked'] == 'S') & 
        (copy_train['Pclass'] == 1)][['Embarked', 'Pclass', 'Ticket']]

#%%%
copy_train[(copy_train['Embarked'] == 'S') & 
        (copy_train['Pclass'] == 2)][['Embarked', 'Pclass', 'Ticket']]

#%%%
copy_train[(copy_train['Embarked'] == 'S') & 
        (copy_train['Pclass'] == 3)][['Embarked', 'Pclass', 'Ticket']]

#%%%
copy_train[(copy_train['Embarked'] == 'C') & 
        (copy_train['Pclass'] == 1)][['Embarked', 'Pclass', 'Ticket']]

#%%%
copy_train[(copy_train['Embarked'] == 'C') & 
        (copy_train['Pclass'] == 2)][['Embarked', 'Pclass', 'Ticket']]

#%%%
copy_train[(copy_train['Embarked'] == 'C') & 
        (copy_train['Pclass'] == 3)][['Embarked', 'Pclass', 'Ticket']]

#%%%
copy_train[(copy_train['Embarked'] == 'Q') & 
        (copy_train['Pclass'] == 1)][['Embarked', 'Pclass', 'Ticket']]

#%%%
copy_train[(copy_train['Embarked'] == 'Q') & 
        (copy_train['Pclass'] == 2)][['Embarked', 'Pclass', 'Ticket']]

#%%%
copy_train[(copy_train['Embarked'] == 'Q') & 
        (copy_train['Pclass'] == 3)][['Embarked', 'Pclass', 'Ticket']]

#%%%
## Fill embarked null value
copy_train['Embarked'] = copy_train['Embarked'].fillna(copy_train['Embarked'].mode()[0])

#%%%
copy_test['Embarked'] = copy_test['Embarked'].fillna(copy_test['Embarked'].mode()[0])

#%%%
## Visualize embarked and Pclass
fig, ax = plt.subplots(figsize = (10,8))
ax = sns.countplot(x = 'Embarked', hue = 'Pclass',data = copy_train)
ax.legend()
plt.show()
plt.close()

#%%%
## Visualize embarked & Survived
fig, ax = plt.subplots(figsize = (10,8))
ax = sns.countplot(x = 'Embarked', hue = 'Survived', data = copy_train)
ax.legend()
plt.show()
plt.close()

#%%%
fig = plt.subplot()
grid = sns.FacetGrid(data = copy_train, col = 'Embarked', 
                     row = 'Pclass', aspect = 1.6)
grid.map(sns.countplot, 'Survived', order = [0, 1])
grid.add_legend()
plt.show()
plt.close()

#%%
## Travel alone or not

copy_train.loc[copy_train['Family'] > 0, 'Tr_alone'] = 0
copy_train.loc[copy_train['Family'] == 0, 'Tr_alone'] = 1

#%%%
## Visualize the survival with travel alone
fig, ax = plt.subplots(figsize = (10,6))
ax = sns.countplot(x = 'Survived', hue = 'Tr_alone', data = copy_train)
plt.show()
plt.close()

#%%%
## Visualize the survival with family member
fig, ax = plt.subplots(figsize = (10,6))
ax = sns.countplot(x = 'Family', hue = 'Survived', data = copy_train)
plt.show()
plt.close()

#%%%
copy_test.loc[copy_test['Family'] > 0, 'Tr_alone'] = 0
copy_test.loc[copy_test['Family'] == 0, 'Tr_alone'] = 1

#%%%
copy_train = pd.concat([copy_train, pd.get_dummies(copy_train['Embarked'],
                                                   prefix = 'Emb')], axis = 1)
copy_test = pd.concat([copy_test, pd.get_dummies(copy_test['Embarked'],
                                                 prefix = 'Emb')], axis = 1)

#%%%
copy_train['N_Fare'] = copy_train['Fare']/ (copy_train['Family']+1)
copy_test['N_Fare'] = copy_test['Fare']/ (copy_test['Family']+1)

#%%%
del_cols = ['Name', 'Ticket', 'Fare', 'SibSp', 'Parch', 'Cabin', 'Embarked',
            'Title', 'PassengerId', 'Family']

#%%%
copy_train.drop(del_cols, axis = 1, inplace = True)
copy_test.drop(del_cols, axis = 1, inplace = True)

#%%%
copy_train.groupby('Pclass')['N_Fare'].mean()

#%%%
# Run models

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import *
from sklearn.model_selection import GridSearchCV
#xgb
#lightgb

#%%
copy_train['Sex'] = LabelEncoder().fit_transform(copy_train['Sex'])
copy_test['Sex'] = LabelEncoder().fit_transform(copy_test['Sex'])

#%%%
X = copy_train.loc[:, copy_train.columns != 'Survived']
Y = copy_train['Survived']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = .20,
                                                    random_state= 42)

#%%
## All models
sgd = SGDClassifier(max_iter = 30, tol = None)
rf = RandomForestClassifier(n_estimators = 100)
knn = KNeighborsClassifier(n_neighbors = 3)
gnb = GaussianNB()
svc = SVC()
dc = DecisionTreeClassifier()
gb = GradientBoostingClassifier()

#%%%
models = [sgd, rf, knn, gnb, svc, dc, gb]

results = pd.DataFrame()

for model in models:
    temp = {}
    model.fit(x_train, y_train)
    model_name = str(model).split('(')[0]
    score = round(model.score(x_train, y_train)*100, 2)
    
    temp['model'] = model_name
    temp['score'] = score
    
    print('The model %s has scored %.2f' % (model_name,score))
    #print('The model {} has scored {:.2f}'.format(model_name, score))
    results = results.append(temp, ignore_index= True)

print(results)

## Loooks like Random Forest and Decision Tree has earned highest score
#%%
## Diving more into the model useing seting hyperparameters and crossfold

sgd_params = {'loss': ['hinge', 'log', 'modified_huber'],
              'penalty': ['l1', 'l2', 'elasticnet'],
              'n_jobs': [-1],
              'early_stopping': [True]}

rf_params = {'n_estimators': [100, 1000, 1000],
             'criterion': ['gini', 'entropy'],
             'oob_score': [False, True],
             'n_jobs': [-1]}

knn_params = {'n_neighbors': [3, 6, 7],
              'leaf_size:': [10, 30, 50],
              'n_jobs': [-1]}

gnb_params = {}

svc_params = {'kernel': ['rbf', 'sigmoid'],
              'random_state': [42]}

dc_params = {'criterion': ['gini', 'entropy'],
             'max_depth': [3, 5],
             'min_samples_split': [2, 10, 50, 100]}

gb_paramas = {'n_estimators': [100, 1000, 10000],
              'criterion': ['friedman_mse', 'mse', 'mae'],
              'min_samples_split': [2, 10, 50, 100],
              'random_state': [42]}

#xgb_params = {'n_estimators':[100, 1000, 10000],
#              'random_state': [42],
#              'n_jobs': [-1]}

#lgb_params = {'n_estimators':[100, 1000, 10000],
#              'random_state': [42],
#              'n_jobs': [-1]}

#%%
# Check one
cv = 3
clf = GridSearchCV(estimator = rf, param_grid = rf_params, cv =cv)
clf.fit(x_train, y_train)

clf.cv_results_
clf.best_estimator_
clf.best_score_
clf.best_params_
#%%%

