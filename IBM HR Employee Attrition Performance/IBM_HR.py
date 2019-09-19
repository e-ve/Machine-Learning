#Importing the Libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

from sklearn.svm import SVC,NuSVC
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler	
from sklearn.metrics import confusion_matrix

LABELS = ["No Attrition", "Attrition"]


#Suppress warnings
import warnings
warnings.filterwarnings('ignore')


#Import Employee Attrition data
data=pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
data.head()
#check for null value
data.info()

data.isnull().values.any()
#No NAN Values 
#Good to Go next step
data.describe()

"""Standard deviation(std) for the fields "EmployeeCount" and ."StandardHours" 
are ZERO. Hence these fields does not add value, hence they can be removed.
Also OVer18 is same for all field .So can be dropped
"""
#These fields does not add value, hence removed
data = data.drop(['EmployeeCount','Over18' , 'StandardHours'], axis = 1)

#A lambda function is a small anonymous function.
#A lambda function can take any number of arguments, but can only have one expression.
#Below , conversion done for Attrition
# Yes is 1
#No is 0
data['Attrition']=data['Attrition'].apply(lambda x : 1 if x=='Yes' else 0)



#Data Balancing Checking
count_classes = pd.value_counts(data['Attrition'])
count_classes.plot(kind = 'bar', rot=0)
plt.title("IBM Attrition Data")
plt.xticks(range(2), LABELS)
plt.xlabel("No Attrition / Attrition")
plt.ylabel("Number of Employee")
plt.show()

"""get correlations of each features in dataset
By plotting a correlation matrix, 
we have a very nice overview of how the features are related to one another
"""
correlationData = data.corr()
top_corr_features = correlationData.index
plt.figure(figsize=(20,20))
#plot heat map
CMAP="RdYlGn"
g=sns.heatmap(data[top_corr_features].corr(),cmap=CMAP)

"""
From the correlation plots, we can see that quite a lot of our columns seem to be poorly correlated with one another.
Generally when making a predictive model, it would be preferable to train a model with features that are not too correlated 
with one another so that we do not need to deal with redundant features. 
In the case that we have quite a lot of correlated features one could perhaps apply a technique such as Principal Component Analysis (PCA) to reduce the feature space."""

#This function is used to convert Categorical values to Numerical values
data=pd.get_dummies(data)

#Separating Feature and Target matrices
X_Result = data.drop(['Attrition'], axis=1)
y_Result=data['Attrition']


#Feature scaling is a method used to standardize the range of independent variables or features of data.
#Since the range of values of raw data varies widely, in some machine learning algorithms, objective functions will not work properly without normalization. 

scale = StandardScaler()
X_Result = scale.fit_transform(X_Result)

# Split the data into Training set and Testing set

X_train, X_test, y_train, y_test = train_test_split(X_Result,y_Result,test_size =0.2,random_state=42)



#Function to Train and Test Machine Learning Model
def train_test_ml_model(X_train,y_train,X_test,Model):
    model.fit(X_train,y_train) #Train the Model
    y_pred = model.predict(X_test) #Use the Model for prediction

    # Test the Model
   
    cm = confusion_matrix(y_test,y_pred)
    accuracy = round(100*np.trace(cm)/np.sum(cm),1)

    print(cm)
    print('Accuracy of the Model' ,Model, str(accuracy)+'%')
    


while True:
    print("\nEnter Number to see the Model Results\n(1)SVC Model\n(2)NuSVC Model\n(3)XGBoost Model\n(4)KNN Model\n(5)Gaussian Model\n(6)Logistic Model\n(7)Desicion Model\n(8)RandomForest Model\n(9)Adaboost Model\n(10)Gradient Model\n(0)Quit")
    choice = int(input(">>> "))
    print(choice)
    if choice==0:
        break
    elif choice==1:
        Model = "SVC"
        model=SVC()
        train_test_ml_model(X_train,y_train,X_test,Model)

    elif choice==2:
        Model = "NuSVC"
        model=NuSVC(nu=0.285)
        train_test_ml_model(X_train,y_train,X_test,Model)

    elif choice==3:
        Model = "XGBClassifier()"
        model=XGBClassifier()
        train_test_ml_model(X_train,y_train,X_test,Model)

    elif choice==4:
        Model = "KNeighborsClassifier"
        model=KNeighborsClassifier()
        train_test_ml_model(X_train,y_train,X_test,Model)

    elif choice==5:
        Model = "GaussianNB"
        model=GaussianNB()
        train_test_ml_model(X_train,y_train,X_test,Model)

    elif choice==6:
        Model = "LogisticRegression"
        model=LogisticRegression()
        train_test_ml_model(X_train,y_train,X_test,Model)

    elif choice==7:
        Model = "DecisionTreeClassifier"
        model=DecisionTreeClassifier()
        train_test_ml_model(X_train,y_train,X_test,Model)

    elif choice==8:
        Model = "RandomForestClassifier"
        model=RandomForestClassifier()
        train_test_ml_model(X_train,y_train,X_test,Model)

    elif choice==9:
        Model = "AdaBoostClassifier"
        model=AdaBoostClassifier()
        train_test_ml_model(X_train,y_train,X_test,Model)
        
    elif choice==10:
        Model = "GradientBoostingClassifier"
        model=GradientBoostingClassifier()
        train_test_ml_model(X_train,y_train,X_test,Model)
        
    else:
        print("Invalid choice, Ending the Proess\n")
        break


