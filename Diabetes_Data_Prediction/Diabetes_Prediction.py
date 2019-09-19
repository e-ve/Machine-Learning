# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")



# Importing the dataset
dataset = pd.read_csv('diabetes2.csv')

#counting the Datas(Yes or Nos)
yesDiabetes = dataset[dataset['Outcome']==1]
noDiabetes = dataset[dataset['Outcome']==0]
print(yesDiabetes.shape,noDiabetes.shape)

#Showing the Yes/No s
sns.countplot(x='Outcome',data=dataset)
#Corelation Heat Map
plt.figure(figsize=(12,12))
plt.title('Correlation of Features', y=1.05, size=15)
sns.heatmap(dataset.corr(),
            linewidths=0.1,
            vmax=1.0, 
            square=True,
            cmap="YlGnBu",
            linecolor='white',
            annot=True)

X = dataset.iloc[:, 0:8].values
y = dataset.iloc[:, 8].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Applying LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
accuracy = round(100*np.trace(cm)/np.sum(cm),1)
print('Accuracy of the Model', str(accuracy)+'%')

