# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19" _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5" papermill={"duration": 3.469246, "end_time": "2025-01-14T23:52:08.574416", "exception": false, "start_time": "2025-01-14T23:52:05.105170", "status": "completed"}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as stats

# %% papermill={"duration": 0.056346, "end_time": "2025-01-14T23:52:08.636415", "exception": false, "start_time": "2025-01-14T23:52:08.580069", "status": "completed"}
df = pd.read_csv('/kaggle/input/titanic/train.csv')

numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = [col for col in df.columns if col not in numerical_cols]

df[numerical_cols].head()

# %% papermill={"duration": 0.02104, "end_time": "2025-01-14T23:52:08.662843", "exception": false, "start_time": "2025-01-14T23:52:08.641803", "status": "completed"}
df[categorical_cols].head()

# %% papermill={"duration": 0.018689, "end_time": "2025-01-14T23:52:08.687170", "exception": false, "start_time": "2025-01-14T23:52:08.668481", "status": "completed"}
df[numerical_cols].isnull().sum()

# %% papermill={"duration": 0.050161, "end_time": "2025-01-14T23:52:08.742825", "exception": false, "start_time": "2025-01-14T23:52:08.692664", "status": "completed"}
imputer = KNNImputer(n_neighbors=2)
imputed_data = imputer.fit_transform(df[numerical_cols])
imputed_df = pd.DataFrame(imputed_data, columns=numerical_cols)
df[numerical_cols] = imputed_df
df[numerical_cols].isnull().sum()

# %% papermill={"duration": 0.051972, "end_time": "2025-01-14T23:52:08.805028", "exception": false, "start_time": "2025-01-14T23:52:08.753056", "status": "completed"}
numerical_cols = [col for col in numerical_cols if col != 'PassengerId']
df = df.drop('PassengerId', axis=1)

df[numerical_cols].head()

# %% papermill={"duration": 0.288029, "end_time": "2025-01-14T23:52:09.107078", "exception": false, "start_time": "2025-01-14T23:52:08.819049", "status": "completed"}
plt.figure(figsize=(8, 6))
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title('Distribution of Pclass with Survived')
plt.xlabel('Pclass')
plt.ylabel('Count')
plt.show()

# %% papermill={"duration": 0.01725, "end_time": "2025-01-14T23:52:09.131069", "exception": false, "start_time": "2025-01-14T23:52:09.113819", "status": "completed"}
df['Age'].skew()

# %% papermill={"duration": 0.33378, "end_time": "2025-01-14T23:52:09.471414", "exception": false, "start_time": "2025-01-14T23:52:09.137634", "status": "completed"}
plt.figure(figsize=(8, 6))
sns.histplot(df['Age'], bins=20, kde=True)
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# %% papermill={"duration": 0.164873, "end_time": "2025-01-14T23:52:09.644716", "exception": false, "start_time": "2025-01-14T23:52:09.479843", "status": "completed"}
plt.figure(figsize=(8, 6))
sns.boxplot(x='Age', data=df)
plt.title('Boxplot of Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# %% papermill={"duration": 0.018384, "end_time": "2025-01-14T23:52:09.670862", "exception": false, "start_time": "2025-01-14T23:52:09.652478", "status": "completed"}
df['Fare'].skew()

# %% papermill={"duration": 0.350629, "end_time": "2025-01-14T23:52:10.029182", "exception": false, "start_time": "2025-01-14T23:52:09.678553", "status": "completed"}
plt.figure(figsize=(8, 6))
sns.histplot(df['Fare'], bins=20, kde=True)
plt.title('Histogram of Fare')
plt.xlabel('Fare')
plt.ylabel('Count')
plt.show()

# %% papermill={"duration": 0.213035, "end_time": "2025-01-14T23:52:10.255392", "exception": false, "start_time": "2025-01-14T23:52:10.042357", "status": "completed"}
plt.figure(figsize=(8, 6))
sns.boxplot(x='Fare', data=df)
plt.title('Boxplot of Fare')
plt.xlabel('Fare')
plt.ylabel('Count')
plt.show()

# %% papermill={"duration": 0.029986, "end_time": "2025-01-14T23:52:10.299612", "exception": false, "start_time": "2025-01-14T23:52:10.269626", "status": "completed"}
df[df['Fare']>=450]

# %% papermill={"duration": 0.021683, "end_time": "2025-01-14T23:52:10.330151", "exception": false, "start_time": "2025-01-14T23:52:10.308468", "status": "completed"}
df['Fare'] = np.cbrt(df['Fare'])
df['Fare'].skew() 

# %% papermill={"duration": 0.352005, "end_time": "2025-01-14T23:52:10.691376", "exception": false, "start_time": "2025-01-14T23:52:10.339371", "status": "completed"}
plt.figure(figsize=(8, 6))
sns.histplot(df['Fare'], bins=20, kde=True)
plt.title('Histogram of Fare')
plt.xlabel('Fare')
plt.ylabel('Count')
plt.show()

# %% papermill={"duration": 0.206756, "end_time": "2025-01-14T23:52:10.907752", "exception": false, "start_time": "2025-01-14T23:52:10.700996", "status": "completed"}
plt.figure(figsize=(8, 6))
sns.boxplot(x='Fare', data=df)
plt.title('Boxplot of Fare')
plt.xlabel('Fare')
plt.ylabel('Count')
plt.show()

# %% papermill={"duration": 0.031297, "end_time": "2025-01-14T23:52:10.949388", "exception": false, "start_time": "2025-01-14T23:52:10.918091", "status": "completed"}
df['Sex'] = df['Sex'].map({
    'male':1,
    'female':0
})

df['Embarked'] = df['Embarked'].map({
    'S':1,
    'C':2,
    'Q':3
})

df[categorical_cols].isnull().sum()

# %% papermill={"duration": 0.032112, "end_time": "2025-01-14T23:52:10.991574", "exception": false, "start_time": "2025-01-14T23:52:10.959462", "status": "completed"}
categorical_cols = [col for col in categorical_cols if col == 'Sex' or col == 'Embarked']
df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)
df[categorical_cols].head()

# %% papermill={"duration": 0.024407, "end_time": "2025-01-14T23:52:11.026471", "exception": false, "start_time": "2025-01-14T23:52:11.002064", "status": "completed"}
df = df.dropna(subset=['Embarked'])
df.isnull().sum()


# %% papermill={"duration": 0.473465, "end_time": "2025-01-14T23:52:11.510686", "exception": false, "start_time": "2025-01-14T23:52:11.037221", "status": "completed"}
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, cmap='coolwarm', fmt='.2f', annot=True, linewidths=.5)
plt.show()

# %% papermill={"duration": 0.032234, "end_time": "2025-01-14T23:52:11.561792", "exception": false, "start_time": "2025-01-14T23:52:11.529558", "status": "completed"}
df = df[['Pclass', 'Sex', 'Fare', 'Survived']]
df.head()
