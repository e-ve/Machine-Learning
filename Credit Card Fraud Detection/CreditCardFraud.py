"""
Data Set Download Link
https://www.kaggle.com/mlg-ulb/creditcardfraud/downloads/creditcardfraud.zip/3
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



LABELS = ["Normal", "Fraud"]


data = pd.read_csv('creditcard.csv')
data.head()



data.info()



data.isnull().values.any()



count_classes = pd.value_counts(data['Class'])
count_classes.plot(kind = 'bar', rot=0)
plt.title("Transaction Class Distribution")
plt.xticks(range(2), LABELS)
plt.xlabel("Normal / Fraud  Class")
plt.ylabel("Number of Transactions")


## Get the Fraud and the normal dataset 

fraud = data[data['Class']==1]

normal = data[data['Class']==0]



print(fraud.shape,normal.shape)



## We need to analyze more amount of information from the transaction data
#How different are the amount of money used in different transaction classes?
fraud.Amount.describe()



normal.Amount.describe()




f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Amount per transaction by class')
bins = 50
ax1.hist(fraud.Amount, bins = bins)
ax1.set_title('Fraud')
ax2.hist(normal.Amount, bins = bins)
ax2.set_title('Normal')
plt.xlabel('Amount ($)')
plt.ylabel('Number of Transactions')
plt.xlim((0, 20000))
plt.yscale('log')
plt.show();




# We Will check Do fraudulent transactions occur more often during certain time frame ? Let us find out with a visual representation.

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Time of transaction vs Amount by class')
ax1.scatter(fraud.Time, fraud.Amount)
ax1.set_title('Fraud')
ax2.scatter(normal.Time, normal.Amount)
ax2.set_title('Normal')
plt.xlabel('Time (in Seconds)')
plt.ylabel('Amount')
plt.show()


#get correlations of each features in dataset
corrmat = data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")




var = data.columns.values

i = 0
t0 = data.loc[data['Class'] == 0]
t1 = data.loc[data['Class'] == 1]

sns.set_style('whitegrid')
plt.figure()
fig, ax = plt.subplots(8,4,figsize=(16,28))

for feature in var:
    i += 1
    plt.subplot(8,4,i)
    sns.kdeplot(t0[feature], bw=0.5,label="Class = 0")
    sns.kdeplot(t1[feature], bw=0.5,label="Class = 1")
    plt.xlabel(feature, fontsize=12)
    locs, labels = plt.xticks()
    plt.tick_params(axis='both', which='major', labelsize=12)
plt.show();


# 
# 
# For some of the features we can observe a good selectivity in terms of distribution for the two values of Class: V4, V11 have clearly separated distributions for Class values 0 and 1, V12, V14, V18 are partially separated, V1, V2, V3, V10 have a quite distinct profile, whilst V25, V26, V28 have similar profiles for the two values of Class.
# 
# In general, with just few exceptions (Time and Amount), the features distribution for legitimate transactions (values of Class = 0) is centered around 0, sometime with a long queue at one of the extremities. In the same time, the fraudulent transactions (values of Class = 1) have a skewed (asymmetric) distribution.
# 

# # Predictive Model



target  = 'Class'
predictor = ['Time', 'V1', 'V2', 'V3', 
     'V4', 'V5', 'V6', 'V7', 
     'V8', 'V9', 'V10','V11', 
     'V12', 'V13', 'V14', 'V15', 
     'V16', 'V17', 'V18', 'V19',
     'V20', 'V21', 'V22', 'V23', 
     'V24', 'V25', 'V26', 'V27',
     'V28', 'Amount']



## Split data in train, test and validation set
from sklearn.model_selection import train_test_split

data_train, data_test = train_test_split(data, test_size=0.20, random_state=2000,shuffle=True )
data_train, data_valid = train_test_split(data_train, test_size=0.20, random_state=2000,shuffle=True)




#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier as RFC
rfc = RFC(n_jobs=4, 
          random_state=2000,
          criterion='gini',
          n_estimators=100,
          verbose=False)




rfc.fit(data_train[predictor], data_train[target].values)




predsRFC = rfc.predict(data_valid[predictor])




from sklearn.metrics import confusion_matrix
cmRFC = confusion_matrix(data_valid[target].values, predsRFC)

cmRFC

from sklearn.metrics import roc_auc_score
accuracyRFC=roc_auc_score(data_valid[target].values, predsRFC)
print("The accuracy of Random Forest Classification is: " + str(accuracyRFC*100)+"%")
