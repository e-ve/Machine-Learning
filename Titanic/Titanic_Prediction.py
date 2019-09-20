import warnings
warnings.simplefilter("ignore")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import re as re
import collections as col

LABELS = ["Not Survived", "Survived"]
# Importing the dataset from 2 CSV files for Train and Test 
train_ds = pd.read_csv('train.csv')
test_ds = pd.read_csv('test.csv')
full_data=[train_ds,test_ds]
#head
train_ds.head()
test_ds.head()

train_ds.info()
""" ********************Feature Engineering******************** """

#P Class
train_ds[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()
#Sex
train_ds[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean()
#SibSp and Parch
#We can creat a Family from Sibling , Childerns
#Also we should check if there are some Single travellsers or Not
for data in full_data:
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
train_ds[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean()

for data in full_data:
    data['IsAlone'] = 0
    data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1
train_ds[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()

#Embarked
#It has some missing Values
train_ds['Embarked'].value_counts()
#As It has maximum number of S , we will fill the data with S
for data in full_data:
    data['Embarked'] = data['Embarked'].fillna('S')
train_ds[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean()

#Fare
#It has Missing Values
#We will Catagorize it

for data in full_data:
    data['Fare'] = data['Fare'].fillna(train_ds['Fare'].median())
train_ds['CategoricalFare'] = pd.qcut(train_ds['Fare'], 4)
train_ds[['CategoricalFare', 'Survived']].groupby(['CategoricalFare'], as_index=False).mean()

#Age
#It has lots of Missing Values

for data in full_data:
    age_avg 	   = data['Age'].mean()
    age_std 	   = data['Age'].std()
    age_nan_count = data['Age'].isnull().sum()
    
    age_nan_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_nan_count)
    data['Age'][np.isnan(data['Age'])] = age_nan_random_list
    data['Age'] = data['Age'].astype(int)
    
train_ds['CategoricalAge'] = pd.cut(train_ds['Age'], 5)

train_ds[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).mean()

#Name
def get_title(name):
	title_search = re.search(' ([A-Za-z]+)\.', name)
	# If the title exists, extract and return it.
	if title_search:
		return title_search.group(1)
	return ""

for data in full_data:
    data['Title'] = data['Name'].apply(get_title)

pd.crosstab(train_ds['Title'], train_ds['Sex'])

#Uniform TItle apply

for data in full_data:
    data['Title'] = data['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    data['Title'] = data['Title'].replace('Mlle', 'Miss')
    data['Title'] = data['Title'].replace('Ms', 'Miss')
    data['Title'] = data['Title'].replace('Mme', 'Mrs')

train_ds[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()



""" ************************Data Cleaning************************ """
for data in full_data:
    # Mapping Sex
    data['Sex'] = data['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    
    # Mapping titles
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    data['Title'] = data['Title'].map(title_mapping)
    data['Title'] = data['Title'].fillna(0)
    
    # Mapping Embarked
    data['Embarked'] = data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    
    # Mapping Fare
    data.loc[ data['Fare'] <= 7.91, 'Fare'] 				           = 0
    data.loc[(data['Fare'] > 7.91) & (data['Fare'] <= 14.454), 'Fare'] = 1
    data.loc[(data['Fare'] > 14.454) & (data['Fare'] <= 31), 'Fare']   = 2
    data.loc[ data['Fare'] > 31, 'Fare'] 							   = 3
    data['Fare'] = data['Fare'].astype(int)
    
    # Mapping Age
    data.loc[ data['Age'] <= 16, 'Age'] 					  = 0
    data.loc[(data['Age'] > 16) & (data['Age'] <= 32), 'Age'] = 1
    data.loc[(data['Age'] > 32) & (data['Age'] <= 48), 'Age'] = 2
    data.loc[(data['Age'] > 48) & (data['Age'] <= 64), 'Age'] = 3
    data.loc[ data['Age'] > 64, 'Age']                        = 4

# Feature Selection
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp',\
                 'Parch', 'FamilySize']
train_ds = train_ds.drop(drop_elements, axis = 1)
train_ds = train_ds.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)

test_ds  = test_ds.drop(drop_elements, axis = 1)

train_ds.head(10)

train_ds= train_ds.values
test_ds  = test_ds.values

""" ********************Classifier Comparison******************** """


import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

classifiers = [
    KNeighborsClassifier(3),
    SVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
	AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    LogisticRegression()]

log_cols = ["Classifier", "Accuracy"]
log 	 = pd.DataFrame(columns=log_cols)

sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)

X = train_ds[0::, 1::]
y = train_ds[0::, 0]

acc_dict = {}

for train_index, test_index in sss.split(X, y):
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]
	
	for clf in classifiers:
		name = clf.__class__.__name__
		clf.fit(X_train, y_train)
		train_predictions = clf.predict(X_test)
		acc = accuracy_score(y_test, train_predictions)
		if name in acc_dict:
			acc_dict[name] += acc
		else:
			acc_dict[name] = acc

for clf in acc_dict:
	acc_dict[clf] = acc_dict[clf] / 10.0
	log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)
	log = log.append(log_entry)

plt.xlabel('Accuracy')
plt.title('Classifier Accuracy')
sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")

#SVC has Highest Accuracy
classifier = SVC()
classifier.fit(train_ds[0::, 1::], train_ds[0::, 0])
result =classifier.predict(test_ds)

col.Counter(result)