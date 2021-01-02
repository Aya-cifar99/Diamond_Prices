#!/usr/bin/env python
# coding: utf-8

# ## 1.To take a look at the big picture  

# our model should learn from classic dataset contains the prices and other attributes of almost 54,000 diamonds ,and be able to predict the diamond price .
# Features :
# 
# price : price in US dollars (\$326--\$18,823)
# 
# carat: weight of the diamond (0.2--5.01)
# 
# cut :quality of the cut (Fair, Good, Very Good, Premium, Ideal)
# 
# color :diamond color, from J (worst) to D (best)
# 
# clarity: a measurement of how clear the diamond is (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best))
# 
# x :length in mm (0--10.74)
# 
# y :width in mm (0--58.9)
# 
# z :depth in mm (0--31.8)
# 
# depth :total depth percentage = z / mean(x, y) = 2 * z / (x + y) (43--79)
# 
# table :width of top of diamond relative to widest point (43--95)
# 

# ## 2.Get the data

# In[48]:


import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[49]:


df = pd.read_csv('diamonds.csv')
dimsdf = df.copy()
dims2 = dimsdf.copy()


# In[ ]:





# In[43]:


dimsdf.head()


# In[228]:


dimsdf.info()


# In[229]:


dimsdf.describe()


# In[59]:


dimsdf.hist(bins=40,figsize=(20,15))


# In[60]:


sns.kdeplot(dimsdf['price']) #skwed to the right 


# In[62]:


sns.kdeplot(dimsdf['depth'])#bell-shaped 


# ## 3.Discover and visulize the data to get insight 

# ### Looking for correlations

# In[30]:


sns.pairplot(dimsdf)


# In[11]:


dimsdf.corr()


# In[66]:


#correlation = dimsdf.corr()
#correlation['carat'].sort_values(ascending = False)
# OR 


# In[204]:


#dimsdf.corr()['carat'].sort_values(ascending=False)


# In[74]:


dimsdf.corr()['depth'].sort_values(ascending=False)


# In[75]:



sns.heatmap(dimsdf.corr(), annot=True)


# In[80]:


sns.pairplot(dimsdf,hue='color',x_vars=['carat','price'],y_vars=['price','x','y','z'])


# In[80]:


sns.pairplot(dimsdf,hue='cut',x_vars=['carat','price'],
             y_vars=['price','x','y','z'],
             palette='coolwarm')


# In[205]:


#sns.pairplot(dimsdf,hue='clarity',x_vars=['carat','price'],y_vars=['price','x','y','z'])


# In[13]:


#dimsdf.plot(kind="scatter",x='price',y='z',alpha=0.5) 


# In[207]:


dimsdf.plot.scatter(x='depth',y='table',cmap='coolwarm')


# In[65]:


dimsdf.plot.scatter(x='carat',y='price',c='x',cmap='coolwarm')


# ### Categorical attributes visulization

# In[57]:


sns.boxplot(x='price',y='cut',data=dimsdf)


# In[64]:


sns.boxplot(x='price',y='color',data=dimsdf)


# In[53]:


sns.boxplot(x='clarity',y='price',data=dimsdf)
#It seems that VS1 and VS2 affect the Diamond's Price equally 


# In[62]:


dimsdf['color'].value_counts().plot(kind="bar")


# In[63]:


dimsdf['cut'].value_counts().plot(kind="bar")


# In[139]:


dimsdf['clarity'].value_counts().plot(kind="bar")


# ## 4.Prepare data for machine learning algorithms

# ### Aggregation column and drop 'Unnamed: 0' col

# In[50]:


dimsdf['diamond_size'] = dimsdf['x']*dimsdf['y']*dimsdf['z']
dimsdf.drop(['Unnamed: 0','x','y','z'] , axis=1,inplace=True)


# In[4]:


dimsdf.head()


# ### convert categorical attributes into numerical (custom Encoder )

# In[51]:


def color_switch(arg):
    if arg == 'D':
         return 1
    elif arg == 'E':
         return 2
    elif arg == 'F':
         return 3
    elif arg == 'G':
         return 4
    elif arg == 'H':
        return 5
    elif arg == 'I':
        return 6
    elif arg == 'J':
        return 7
    else:
        return None


# In[52]:


def clarity_switch(arg):
    if arg == 'IF':
         return 1
    elif arg == 'VVS1':
         return 2
    elif arg == 'VVS2':
         return 3
    elif arg == 'VS1':
         return 4
    elif arg == 'VS2':
        return 5
    elif arg == 'SI1':
        return 6
    elif arg == 'SI2':
        return 7
    elif arg == 'I1':
        return 8
    else:
        return None


# In[53]:


def cut_switch(arg):
    if arg == 'Ideal':
         return 1
    elif arg == 'Premium':
         return 2
    elif arg == 'Very Good':
         return 3
    elif arg == 'Good':
         return 4
    elif arg == 'Fair':
        return 5
    else:
        return None


# In[54]:


dimsdf['cut'] = dimsdf['cut'].apply(cut_switch)


# In[55]:


dimsdf['clarity'] = dimsdf['clarity'].apply(clarity_switch)


# In[56]:


dimsdf['color'] = dimsdf['color'].apply(color_switch)


# In[57]:


dimsdf.head()


# ### Split data into train , test set

# In[12]:


dimsdf.columns


# In[58]:


x = dimsdf[['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'diamond_size']]
y = dimsdf['price']
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2)


# ## 5.Select and train a model 

# ### Train the model , prediction and model evaluation 

# ### Linear regression

# In[63]:


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import linear_model
import math

regr = linear_model.LinearRegression()
regr.fit(x_train,y_train)
y_pred = regr.predict(x_test)
print("accuracy: "+ str(regr.score(x_train,y_train)*100) + "%")
print("Mean absolute error: {}".format(mean_absolute_error(y_test,y_pred)))
print("Mean squared error: {}".format(mean_squared_error(y_test,y_pred)))
print("Root Mean squared error: {}".format(math.sqrt(mean_squared_error(y_test,y_pred))))


# ### Ridge regression 

# In[64]:


rig_reg = linear_model.Ridge()
rig_reg.fit(x_train,y_train)
y_pred = rig_reg.predict(x_test)
print("accuracy: "+ str(rig_reg.score(x_train,y_train)*100) + "%")
print("Mean absolute error: {}".format(mean_absolute_error(y_test,y_pred)))
print("Mean squared error: {}".format(mean_squared_error(y_test,y_pred)))
print("Root Mean squared error: {}".format(math.sqrt(mean_squared_error(y_test,y_pred))))


# ### Lasso regression

# In[65]:


las_reg = linear_model.Lasso()
las_reg.fit(x_train,y_train)
y_pred = las_reg.predict(x_test)
print("accuracy: "+ str(las_reg.score(x_train,y_train)*100) + "%")
print("Mean absolute error: {}".format(mean_absolute_error(y_test,y_pred)))
print("Mean squared error: {}".format(mean_squared_error(y_test,y_pred)))
print("Root Mean squared error: {}".format(math.sqrt(mean_squared_error(y_test,y_pred))))


# ### Random forest regressor

# In[66]:


from sklearn.ensemble import RandomForestRegressor
random_reg = RandomForestRegressor()
random_reg.fit(x_train,y_train)
y_pred = random_reg.predict(x_test)
print("accuracy: "+ str(random_reg.score(x_train,y_train)*100) + "%")
print("Mean absolute error: {}".format(mean_absolute_error(y_test,y_pred)))
print("Mean squared error: {}".format(mean_squared_error(y_test,y_pred)))
print("Root Mean squared error: {}".format(math.sqrt(mean_squared_error(y_test,y_pred))))


# ### Delete Outliers :

# In[18]:


dims_out = dimsdf.copy()


# In[20]:


dims_out.head()


# In[21]:


Q1=dims_out['price'].quantile(0.25)
Q3=dims_out['price'].quantile(0.75)
IQR=Q3-Q1
Lower_Whisker = Q1 - 1.5*IQR
Upper_Whisker = Q3 + 1.5*IQR
print(Lower_Whisker, Upper_Whisker)
dims_out = dims_out[dims_out['price']< Upper_Whisker]


# In[22]:


dims_out.shape


# In[23]:


Q1=dims_out['depth'].quantile(0.25)
Q3=dims_out['depth'].quantile(0.75)
IQR=Q3-Q1
Lower_Whisker = Q1 - 1.5*IQR
Upper_Whisker = Q3 + 1.5*IQR
print(Lower_Whisker, Upper_Whisker)
dims_out = dims_out[dims_out['depth']< Upper_Whisker]


# In[24]:


dims_out.shape


# In[25]:


Q1=dims_out['carat'].quantile(0.25)
Q3=dims_out['carat'].quantile(0.75)
IQR=Q3-Q1
Lower_Whisker = Q1 - 1.5*IQR
Upper_Whisker = Q3 + 1.5*IQR
print(Lower_Whisker, Upper_Whisker)
dims_out = dims_out[dims_out['carat']< Upper_Whisker]


# In[26]:


dims_out.shape


# In[27]:


Q1=dims_out['table'].quantile(0.25)
Q3=dims_out['table'].quantile(0.75)
IQR=Q3-Q1
Lower_Whisker = Q1 - 1.5*IQR
Upper_Whisker = Q3 + 1.5*IQR
print(Lower_Whisker, Upper_Whisker)
dims_out = dims_out[dims_out['table']< Upper_Whisker]


# In[28]:


dims_out.shape


# In[29]:


Q1=dims_out['diamond_size'].quantile(0.25)
Q3=dims_out['diamond_size'].quantile(0.75)
IQR=Q3-Q1
Lower_Whisker = Q1 - 1.5*IQR
Upper_Whisker = Q3 + 1.5*IQR
print(Lower_Whisker, Upper_Whisker)
dims_out = dims_out[dims_out['diamond_size']< Upper_Whisker]


# In[30]:


dims_out.shape


# ### split data
# 

# In[32]:


x2 = dims_out[['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'diamond_size']]
y2 = dims_out['price']
from sklearn.model_selection import train_test_split
x_train2 , x_test2 , y_train2 , y_test2 = train_test_split(x2,y2,test_size=0.2)


# ### Train model after drop outliers 

# ### Linear regression 

# In[33]:


regr = linear_model.LinearRegression()
regr.fit(x_train2,y_train2)
y_pred = regr.predict(x_test2)
print("accuracy: "+ str(regr.score(x_train2,y_train2)*100) + "%")
print("Mean absolute error: {}".format(mean_absolute_error(y_test2,y_pred)))
print("Mean squared error: {}".format(mean_squared_error(y_test2,y_pred)))


# ### Ridge regression 

# In[34]:


rig_reg = linear_model.Ridge()
rig_reg.fit(x_train2,y_train2)
y_pred = rig_reg.predict(x_test2)
print("accuracy: "+ str(rig_reg.score(x_train2,y_train2)*100) + "%")
print("Mean absolute error: {}".format(mean_absolute_error(y_test2,y_pred)))
print("Mean squared error: {}".format(mean_squared_error(y_test2,y_pred)))


# ### Lasso regression

# In[35]:


las_reg = linear_model.Lasso()
las_reg.fit(x_train2,y_train2)
y_pred = las_reg.predict(x_test2)
print("accuracy: "+ str(las_reg.score(x_train2,y_train2)*100) + "%")
print("Mean absolute error: {}".format(mean_absolute_error(y_test2,y_pred)))
print("Mean squared error: {}".format(mean_squared_error(y_test2,y_pred)))


# ### Random forest regressor

# In[36]:


random_reg = RandomForestRegressor()
random_reg.fit(x_train2,y_train2)
y_pred = random_reg.predict(x_test2)
print("accuracy: "+ str(random_reg.score(x_train2,y_train2)*100) + "%")
print("Mean absolute error: {}".format(mean_absolute_error(y_test2,y_pred)))
print("Mean squared error: {}".format(mean_squared_error(y_test2,y_pred)))

