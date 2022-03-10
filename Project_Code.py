#!/usr/bin/env python
# coding: utf-8

# In[18]:


get_ipython().run_line_magic('config', 'Completer.use_jedi = False')
import numpy as np    
import pandas as pd    
import matplotlib.pyplot as plt   
from sklearn.preprocessing import PolynomialFeatures    
from sklearn.linear_model import LinearRegression, HuberRegressor    
from sklearn.metrics import mean_squared_error    


# In[19]:


FMIRawData = pd.read_csv('ML project_dataset.csv')


# In[20]:


# Check the dataset
FMIRawData.head(5) 


# In[23]:


FMIRawData.columns


# In[24]:


data = FMIRawData.drop(['Time zone'],axis=1)


# In[25]:


# remove the column 'Time zone'
data.head(5)


# In[26]:


data.columns=['Year','Month', 'Day', 'Time','Monthly precipitation amount (mm)']


# In[27]:


data.head(5)


# In[28]:


date_column = data["Year"].astype(str)+'-'+data["Month"].astype(str)+'-'+data["Day"].astype(str)
data.insert(0,"Date",date_column)


# In[29]:


data.head(5)


# In[30]:


newdata = data.loc[data["Year"] == 2002]
print("First five rows of the dataframe 'newdata'\n",newdata.head())


# In[32]:


X = newdata["Month"].to_numpy().reshape(-1, 1)
y = newdata["Monthly precipitation amount (mm)"].to_numpy()


# In[33]:


fig, axes = plt.subplots(1, 2, figsize=(14,5))
axes[0].scatter(newdata['Month'],newdata['Monthly precipitation amount (mm)'])
axes[0].set_xlabel("Month",size=15)
axes[0].set_ylabel("Monthly precipitation amount (mm)",size=15)
axes[0].set_title("2002: Month vs Monthly precipitation amount (mm)",size=15)

axes[1].hist(data['Monthly precipitation amount (mm)'])
axes[1].set_title('distribution of maxMonthly precipitation amount',size=15)
axes[1].set_ylabel("count of datapoints",size=15)
axes[1].set_xlabel("Monthly precipitation amount intervals",size=15)

plt.show()


# In[34]:


regr = LinearRegression()
regr.fit(X,y)


# In[35]:


y_pred = regr.predict(X)
tr_error = mean_squared_error(y,y_pred)


# In[36]:


print('The training error is: ', tr_error)    
print("w1 = ", regr.coef_)   
print("w0 = ",regr.intercept_) 


# In[37]:


print("Because the training error of the linear regrassion is extremely large.")
print("Linear regression is not suitable for applying in this case.")


# In[38]:


# Polynomial regression
degrees = [3, 5, 7, 9, 11, 14]    
tr_errors = []          

for i in range(len(degrees)):   
    
    print("Polynomial degree = ",degrees[i])

    poly = PolynomialFeatures(degree=degrees[i])
    X_poly = poly.fit_transform(X,y)
    
    lin_regr = LinearRegression(fit_intercept=False) 
    lin_regr.fit(X_poly,y)
    
    y_pred = lin_regr.predict(X_poly)
    tr_error = mean_squared_error(y,y_pred)
    tr_errors.append(tr_error)
    X_fit = np.linspace(1, 12, 12)
 
    plt.plot(X_fit, lin_regr.predict(poly.transform(X_fit.reshape(-1, 1))), label="Model")  
    plt.scatter(X, y, color="b", s=10, label="datapoints from the dataframe 'newdata'")  
    plt.xlabel('Month')    
    plt.ylabel('Monthly precipitation amount (mm)')
    plt.legend(loc="best")    
    plt.title('Polynomial degree = {}\nTraining error = {:.5}'.format(degrees[i], tr_error))
    plt.show()    


# In[ ]:




