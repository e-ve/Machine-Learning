# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Startups50.csv')
dataset.head()
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

#Building optimal model using Backward elinimation
#import statsmodels.formula.api as sm

import statsmodels.regression.linear_model as sm
X=np.append(arr = np.ones((50,1)).astype(int) , values=X,axis=1)
X_opt=X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS=sm.OLS(endog=y , exog=X_opt).fit()
regressor_OLS.summary()

#x2 has highest P value, so we discard this
X_opt=X[:, [0, 1, 3, 4, 5]]
regressor_OLS=sm.OLS(endog=y , exog=X_opt).fit()
regressor_OLS.summary()

X_opt=X[:, [0, 3, 4, 5]]
regressor_OLS=sm.OLS(endog=y , exog=X_opt).fit()
regressor_OLS.summary()

X_opt=X[:, [0, 3, 5]]
regressor_OLS=sm.OLS(endog=y , exog=X_opt).fit()
regressor_OLS.summary()

X_opt=X[:, [0, 3,]]
regressor_OLS=sm.OLS(endog=y , exog=X_opt).fit()
regressor_OLS.summary()

'''
import statsmodels.regression.linear_model as sm
def backwardElimination(x, SL):
    numVars = len(x[0])
    temp = np.zeros((50,6)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
        regressor_OLS.summary()
        return x
    
SL = 0.12
#X=np.append(arr = np.ones((50,1)).astype(int) , values=X,axis=1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)

'''