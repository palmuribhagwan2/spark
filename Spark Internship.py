#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


# here we load the data
df=pd.read_csv('https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv')


# In[3]:


df.head(3)


# ## Given Question is:

# ![Screenshot%20%28131%29.png](attachment:Screenshot%20%28131%29.png)

# ## Busines Problem Understanding
# - here dependent variable is percentage of an student(y) i.e. score and no. of study hours is independent variable(x)

# ## Data understanding

# In[4]:


df.shape


# In[5]:


df.info()


# ## EDA

# In[6]:


df.describe()


# In[7]:


sns.pairplot(df)
plt.show()


# ### data cleaning

# In[8]:


df.isnull().sum()   # there is no missing values


# In[9]:


df.skew()   


# In[10]:


sns.displot(x=df['Hours'],kde=True)


# **data is normal because**

# In[11]:


sns.scatterplot(data=df,x='Hours',y='Scores')
plt.grid()
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')


# In[12]:


df.corr()


# In[ ]:





# **From the graph and correlation between variables, we can clearly see that there is a positive linear relation between the number of hours studied and percentage of score.**

# ## Train | Test

# In[13]:


x=df.drop(columns=['Scores'])
x.head(4)


# In[14]:


y=df['Scores']
y.head()


# # from sklearn.model_selection import train_test_split  
# 

# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25, random_state=42) 


# In[17]:


print(x_train.head(3))


# In[18]:


print(x_test.head(2))


# In[19]:


print(y_train.head(3))


# In[20]:


print(y_test.head(3))


# ## Training Algorithm
# - train our algorithm

# In[21]:


# import model
from sklearn.linear_model import LinearRegression

#save model
regressor = LinearRegression()  

# fit_model
regressor.fit(x_train, y_train) 


# In[22]:


#plotting the regression line
line= regressor.coef_*x+ regressor.intercept_
line.head(3)   # this is predited one


# In[23]:


plt.scatter(x,y)
plt.plot(x,line)


# ## Now making prediction

# In[24]:


print(x_test) # Testing data - In Hours
y_pred = regressor.predict(x_test) #(test predictions) Predicting the scores while putting hours
print(pd.DataFrame(y_pred).head(3))


# In[25]:


# train prediction
train_predictions=regressor.predict(x_train)


# In[26]:


# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 


# ## Evaluation

# In[27]:


from sklearn.metrics import mean_squared_error
print('MSE for test',mean_squared_error(y_test,y_pred))
print('MSE for train',mean_squared_error(y_train,train_predictions))


# In[28]:


print('RMSE for test',np.sqrt(mean_squared_error(y_test,y_pred)))
print('RMSE for train',np.sqrt(mean_squared_error(y_train,train_predictions)))


# In[29]:


from sklearn.metrics import r2_score
print('r2 for test data',r2_score(y_test,y_pred))
print('r2 for test data',r2_score(y_train,train_predictions))


# #### r2_score is nearly equal to 1 and both are nearly same, it means our model is good

# ## Prediction

# In[30]:


hrs=9.25
own_prediction=regressor.predict([[hrs]])
own_prediction


# In[31]:


print("No of Hours = {}".format(hrs))
print("Predicted Score = {}".format(own_prediction[0]))


# ### The predicted score is 92.3357 when student's number of study will 9.25 hrs

# In[ ]:




