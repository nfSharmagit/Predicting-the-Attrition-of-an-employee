
# coding: utf-8

# Human Resource Analytics 
# 
# 
# Reference: 
# https://www.kaggle.com/ludobenistant/hr-analytics/data
# 
# Objective:
# 
# Objective is to analyze the data and predict which valuable employees will leave next.
# 
# 
# Fields in the dataset include:
# 
# Satisfaction Level
# 
# Last evaluation
# 
# Number of projects
# 
# Average monthly hours
# 
# Time spent at the company
# 
# Whether they have had a work accident
# 
# Whether they have had a promotion in the last 5 years
# 
# Departments (column sales)
# 
# Salary
# 
# Whether the employee has left

# In[1]:

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

get_ipython().magic('matplotlib inline')
from sklearn.cross_validation import train_test_split


# In[2]:

#Import Csv 
data=pd.read_csv("HR_comma_sep.csv")
data.head()


# In[3]:

#Dataset Info
data.info()


# In[4]:

#Datatypes 
data.dtypes


# Sales and Salary attributes are object. To fit machine learning models using dataset,we first need to extract categorical variables and convert them to numeric variables.
# "pd.get_dummies" helps us in this task.

# In[5]:

numeric_data_sales = pd.get_dummies(data['sales'], prefix='sales')
del data['sales'] 


# In[6]:

numeric_data = data.join(numeric_data_sales)


# In[7]:

numeric_data_salary = pd.get_dummies(numeric_data['salary'], prefix='salary')
del numeric_data['salary'] 


# In[8]:

data_final = numeric_data.join(numeric_data_salary)


# In[9]:

data_final.head()


# In[10]:

data_final.dtypes


# In[11]:

X = data_final.drop('left', axis=1)
y = data_final['left']


# We need to find out which independent variables contributes more towards the Attrition.
# We can use RFE(Recursive Feature Elimination) function to determine important contributing variables.

# In[12]:

logreg = LogisticRegression()
rfe = RFE(logreg,15)
rfe = rfe.fit(X,y )


# In[13]:

#The independent variables that contribute more are stated by True
print(rfe.support_)


# In[14]:

print(rfe.ranking_)


# In[15]:

X.head()


# In[16]:

Data_vars=['satisfaction_level','last_evaluation','number_project','time_spend_company','Work_accident','promotion_last_5years','sales_RandD','sales_accounting','sales_hr','sales_management','sales_support','sales_technical','salary_high','salary_low','salary_medium']
X=data_final[Data_vars]


# In[17]:

X.head()


# Statistical report

# In[18]:

import statsmodels.api as sm

logit = sm.Logit(y,X)


# In[19]:

result=logit.fit()


# In[20]:

#statistical report 
print(result.summary())


# The null (default) hypothesis is always that each independent variable is having absolutely no effect (has a coefficient of 0) and you are looking for a reason to reject this hypothesis.
# 
# The p-Value for sales_accounting and sales_support is > 0.05,considering 95% confidence interval.
# Therefore, we cannot reject null hytpothesis for these two variables.
# We can thus eliminate these two columns to improve the accuracy of the model.
# 
# Insights:
# Salary play an important role as far as attrition is concerned
# Dept. RandD,HR,Management and Technical departments are statistically significant
# Satisfaction level,number of projects,time spent in a company,promotions and work accidents plays major role in Attrition

# In[21]:

X=X.drop(['sales_accounting','sales_support'],axis=1)  


# Splitting the data to avoid model overfitting. 
# We can also use Cross Validation.

# In[22]:

#Splittimg the data into training and testing set
X_train, X_test, y_train, y_test = train_test_split( X ,y , test_size=0.30, random_state=42)


# ## Model Implementation : Logistic Regression 

# In[23]:

#Initiate Classifier 
lgr=LogisticRegression()


# In[24]:

#Fit the model on training set 
lgr.fit(X_train,np.ravel(y_train))


# In[25]:

y_pred_lgr=lgr.predict(X_test)
y_pred_lgr


# In[26]:

from sklearn.metrics import accuracy_score

a_score_lgr = accuracy_score(y_test,y_pred_lgr)
a_score_lgr=a_score_lgr*100
print("Metric function accuracy for training data :%f" %a_score_lgr)


# In[27]:

from sklearn.metrics import confusion_matrix

confusion_matrix_lgr=confusion_matrix(y_test,y_pred_lgr)
print(confusion_matrix_lgr)


# In[28]:

y_pred_lgr


# In[29]:

#Probablity of next Employee leaving 
y_probablity_lgr =  lgr.predict_proba(X_test)
y_probablity_lgr


# The accuracy of logistic regression model is just fair but not that good. Let us try another prediction model as well

# ## Random Forest Classifier 

# In[30]:

clf=RandomForestClassifier()


# In[31]:

clf.fit(X_train,np.ravel(y_train))


# In[32]:

Y_pred_clf=clf.predict(X_test)
Y_pred_clf


# In[33]:

a_score_clf = accuracy_score(y_test,Y_pred_clf)
a_score_clf=a_score_clf*100

print("Metric function accuracy for training data :%f" %a_score_clf)


# In[34]:

confusion_matrix_clf = confusion_matrix(y_test,Y_pred_clf)
print(confusion_matrix_clf)


# In[35]:

from sklearn.metrics import classification_report

print(classification_report(y_test, Y_pred_clf))


# Random Forest has best accuracy of 98.11%.Hence, Random Forest can be used for predicting the next attrition

# In[36]:

#Probablity of next Employee Leaving
y_probablity =  clf.predict_proba(X_test)


# In[37]:

y_probablity


# In[ ]:




# In[ ]:



