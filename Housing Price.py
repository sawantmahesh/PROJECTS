#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# advanced housing price prediction


# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")

pd.set_option("display.max_rows",200)
pd.set_option("display.max_columns",300)


# In[3]:


# there are two csv files train.csv , test.csv


# In[5]:


train_data=pd.read_csv("train.csv")


# In[6]:


df_train=train_data.copy()
df_train.head()


# In[7]:


df_train.shape


# In[8]:


test_data=pd.read_csv("test.csv")


# In[9]:


df_test=test_data.copy()
df_test.head()


# In[10]:


df_test.shape


# In[11]:


total_rows=1460+1459
total_rows


# In[ ]:


# we will concat the two dataset to do the preprocessing torether


# In[12]:


df_train["train/test"]="train"
df_test["train/test"]="test"


# In[13]:


df_train.head()


# In[14]:


df_test.head()


# In[ ]:


#lets concat the two dataset


# In[15]:


data=pd.concat([df_train,df_test],axis=0)# added the two dataset row-wise


# In[16]:


data.head()


# In[17]:


data.tail()


# In[18]:


data.shape


# In[ ]:


# lets handle the missing value


# In[19]:


data.isnull().sum()/len(data)*100


# In[20]:


sns.heatmap(data.isnull())


# In[21]:


high_null=data.isnull().sum()/len(data)*100
high_missing=high_null[high_null>45]


# In[22]:


high_missing.shape


# In[23]:


high_missing


# In[24]:


high_missing.index


# In[25]:


high_missing=['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature']


# In[26]:


data1=data.drop(high_missing,axis=1)


# In[27]:


data1.shape


# In[28]:


data1.head()


# In[29]:


data1.isnull().sum()/len(data1)*100


# In[30]:


data1.info()


# In[ ]:


# how to identify categorical and numerical variable to handing -- misisng value or pre-processing


# In[31]:


int_data=data1.select_dtypes(exclude=["object"])


# In[32]:


int_data.info()


# In[33]:


int_data.isnull().sum()


# In[ ]:


# suppose we dont want to include " salesprice" in the above list


# In[34]:


numerical_columns=[col for col in data1.columns if (data1[col].dtype=="int64" or data1[col].dtype=="float64") and col!="SalePrice"]


# In[35]:


numerical_columns


# In[36]:


data2=data1.copy()


# In[37]:


data2[numerical_columns]=data2[numerical_columns].fillna(data2[numerical_columns].median())


# In[38]:


data2.info()


# In[ ]:


# find the object datatype -- to handle the missing value


# In[39]:


object_data=data2.select_dtypes(include=["object"])


# In[40]:


object_data.info()


# In[41]:


object_data.columns


# In[42]:


data3=data2.copy()


# In[43]:


data3[object_data.columns]=data3[object_data.columns].fillna(data3[object_data.columns].mode().iloc[0])


# In[44]:


data3.isnull().sum()


# In[45]:


sns.heatmap(data3.isnull(),cmap="coolwarm")


# In[ ]:


# we have handled the misisng value


# In[ ]:


# label encoding -- categorical to numerical conversion


# In[ ]:


# we will apply the one hot encoding approach


# In[46]:


object_dataset=object_data.columns
object_dataset


# In[47]:


data4=pd.get_dummies(data3,columns=object_dataset,drop_first=True)


# In[48]:


data3.shape


# In[49]:


data4.shape


# In[ ]:


# feature scaling we will do lateron


# In[ ]:


#split the train and test dataset as given by the stake holder


# In[ ]:


#train data - 1460 rows
#test data - 1459 rows


# In[50]:


training_data=data4.iloc[:1460,:]
final_test=data4.iloc[1460:,:]


# In[51]:


training_data.shape


# In[52]:


final_test.shape


# In[53]:


training_data.head()


# In[ ]:





# In[54]:


training_data=training_data.drop(["train/test_train"],axis=1)


# In[ ]:





# In[55]:


data5=data4.copy()


# In[56]:


data5.tail()


# In[57]:


data10=data5[data5["train/test_train"]==1]


# In[58]:


data10.shape


# In[59]:


data20=data5[data5["train/test_train"]==0]


# In[60]:


data20.shape


# In[ ]:





# In[ ]:


# training_data--
#final_test--


# In[61]:


training_data.head()


# In[62]:


final_test.head()


# In[63]:


final_test=final_test.drop(["SalePrice","train/test_train"],axis=1)


# In[64]:


final_test.head()


# In[65]:


final_test.shape


# In[66]:


training_data.shape


# In[67]:


training_data.head()


# In[ ]:


# feature engineering(scaling)


# In[ ]:


# training dataset


# In[68]:


X=training_data.drop(["Id","SalePrice"],axis=1)
Y=training_data["SalePrice"]


# In[69]:


from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler


# In[70]:


scaler=MinMaxScaler()
X_scaled=scaler.fit_transform(X)
X_scaled=pd.DataFrame(X_scaled,columns=X.columns)
X_scaled.head()


# In[ ]:


# scaling on testing data_set also


# In[71]:


final_test1=final_test.drop(["Id"],axis=1)


# In[72]:


final_test_data=scaler.fit_transform(final_test1)
final_test_data=pd.DataFrame(final_test_data,columns=final_test1.columns)
final_test_data.head()


# In[ ]:


# our final_test_data is ready -- we completed all pre-processing , later we will use it 


# In[ ]:


# lets start model building 


# In[ ]:


#X_scaled--> independent variable
#Y--> dependent variable


# In[ ]:


# lets split the dataset into training and evalaution


# In[75]:


from sklearn.model_selection import train_test_split


# In[76]:


X_train,X_eval,Y_train,Y_eval=train_test_split(X_scaled,Y,test_size=0.2,random_state=100)


# In[77]:


# lets import the algorithms to build the model


# In[78]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error


# In[79]:


lr=LinearRegression()
lr.fit(X_train,Y_train)


# In[80]:


Y_pred=lr.predict(X_eval)


# In[81]:


Y_pred_train=lr.predict(X_train)


# In[82]:


print("training_score",r2_score(Y_train,Y_pred_train))
print("evaluation_score",r2_score(Y_eval,Y_pred))


# R2=1-(RSS/TSS) where RSS=Sum of squared residuals TSS=Total sum of squares
# 
# R2 score can be -ve also.

# In[ ]:


# what is Regularization /optimization.?


# In[ ]:


Regularization technique used to solve the overfitting problem
L1 regularization-- Lasso--- penalty is the sum of absolute values of weight

L2---- ridge----penalty is the sum of square values of weight
lembda is the tuning parameter (regularization parameter)

L1/L2--- Elastic net-- hybrid behavior between L1 and L2 REGULARIZATION


# In[83]:


from sklearn.linear_model import LinearRegression,Ridge,Lasso,RidgeCV,LassoCV,ElasticNet,ElasticNetCV


# In[84]:


lasso=Lasso(alpha=50)


# In[85]:


lasso.fit(X_train,Y_train)


# In[86]:


Y_pred_lasso=lasso.predict(X_eval)


# In[87]:


Y_pred_train=lasso.predict(X_train)


# In[88]:


print("training_score",r2_score(Y_train,Y_pred_train))
print("evaluation_score",r2_score(Y_eval,Y_pred_lasso))


# In[89]:


lasso1=LassoCV(cv=10)
lasso1.fit(X_train,Y_train)
Y_pred_lasso=lasso1.predict(X_eval)
Y_pred_train=lasso1.predict(X_train)
print("training_score",r2_score(Y_train,Y_pred_train))
print("evaluation_score",r2_score(Y_eval,Y_pred_lasso))


# In[90]:


ridge=Ridge(alpha=1)
ridge.fit(X_train,Y_train)
Y_pred_ridge=ridge.predict(X_eval)
Y_pred_train=ridge.predict(X_train)
print("training_score",r2_score(Y_train,Y_pred_train))
print("evaluation_score",r2_score(Y_eval,Y_pred_ridge))


# In[91]:


ridge=RidgeCV(cv=10)
ridge.fit(X_train,Y_train)
Y_pred_ridge=ridge.predict(X_eval)
Y_pred_train=ridge.predict(X_train)
print("training_score",r2_score(Y_train,Y_pred_train))
print("evaluation_score",r2_score(Y_eval,Y_pred_ridge))


# In[92]:


en=ElasticNet()
en.fit(X_train,Y_train)
Y_pred_en=en.predict(X_eval)
Y_pred_train=en.predict(X_train)
print("training_score",r2_score(Y_train,Y_pred_train))
print("evaluation_score",r2_score(Y_eval,Y_pred_en))


# In[93]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR


# In[94]:


from sklearn.ensemble import VotingRegressor,StackingRegressor


# In[95]:


dt=DecisionTreeRegressor()
rf=RandomForestRegressor()
gdb=GradientBoostingRegressor()
ada=AdaBoostRegressor()
knn=KNeighborsRegressor()
svr=SVR()


# In[96]:


for model in[dt,rf,gdb,ada]:
    print("++++++"*5)
    print("performance of", model)
    print("++++++"*5)
    abc=model.fit(X_train,Y_train)
    Y_pred_train=abc.predict(X_train)
    Y_pred_eval=abc.predict(X_eval)
    AS_train=r2_score(Y_train,Y_pred_train)
    AS_eval=r2_score(Y_eval,Y_pred_eval)
    
    print("training accuracy",AS_train)
    #print("++++++"*5)
    print("evaluation accuracy",AS_eval)
    #print("++++++"*5)
    print("mean absolute error",mean_absolute_error(Y_eval,Y_pred_eval))


# In[ ]:


# voting regressor


# In[97]:


estimators=[("dt",DecisionTreeRegressor()),
("rf",RandomForestRegressor()),
("gdb",GradientBoostingRegressor()),
("ada",AdaBoostRegressor()),
("ridge",Ridge())]


# In[98]:


voting=VotingRegressor(estimators)


# In[99]:


voting.fit(X_train,Y_train)


# In[100]:


Y_pred_vote=voting.predict(X_eval)


# In[101]:


r2_score(Y_eval,Y_pred_vote)


# In[ ]:





# In[ ]:





# In[ ]:


#performance of GradientBoostingRegressor is better


# In[ ]:


# final testing and price prediction


# In[102]:


final_test_data.head()


# In[103]:


final_submission=final_test_data.copy()


# In[104]:


Y_final=gdb.predict(final_submission)


# In[105]:


final_submission["predicted_SP"]=Y_final


# In[106]:


final_submission.head()


# In[107]:


final_submission["ID"]=test_data["Id"]


# In[108]:


final_submission=final_submission[["ID","predicted_SP"]]


# In[109]:


final_submission


# In[ ]:




