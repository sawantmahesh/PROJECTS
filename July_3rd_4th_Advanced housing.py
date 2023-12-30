#!/usr/bin/env python
# coding: utf-8

# In[1]:


# advanced housing regression 


# In[2]:


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


# In[4]:


train_data=pd.read_csv("train.csv")


# In[5]:


df_train=train_data.copy()
df_train.head()


# In[6]:


df_train.shape


# In[7]:


test_data=pd.read_csv("test.csv")


# In[8]:


df_test=test_data.copy()
df_test.head()


# In[9]:


df_test.shape


# In[10]:


total_rows=1460+1459
total_rows


# In[11]:


# we will concat the two dataset to do the preprocessing torether


# In[12]:


df_train["train/test"]="train"
df_test["train/test"]="test"


# In[13]:


df_train.head()


# In[14]:


df_test.head()


# In[15]:


#lets concat the two dataset


# In[16]:


data=pd.concat([df_train,df_test],axis=0)# added the two dataset row-wise


# In[17]:


data.head()


# In[18]:


data.tail()


# In[19]:


data.shape


# In[20]:


# lets handle the missing value


# In[21]:


data.isnull().sum()/len(data)*100


# In[22]:


sns.heatmap(data.isnull())


# In[23]:


high_null=data.isnull().sum()/len(data)*100
high_missing=high_null[high_null>45]


# In[24]:


high_missing.shape


# In[25]:


high_missing


# In[26]:


high_missing.index


# In[27]:


high_missing=['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature']


# In[28]:


data1=data.drop(high_missing,axis=1)


# In[29]:


data1.shape


# In[30]:


data1.head()


# In[31]:


data1.isnull().sum()/len(data1)*100


# In[32]:


data1.info()


# In[33]:


# how to identify categorical and numerical variable to handing -- misisng value or pre-processing


# In[34]:


int_data=data1.select_dtypes(exclude=["object"])


# In[35]:


int_data.info()


# In[36]:


int_data.isnull().sum()


# In[37]:


# suppose we dont want to include " salesprice" in the above list


# In[38]:


numerical_columns=[col for col in data1.columns if (data1[col].dtype=="int64" or data1[col].dtype=="float64") and col!="SalePrice"]


# In[39]:


numerical_columns


# In[40]:


data2=data1.copy()


# In[41]:


data2[numerical_columns]=data2[numerical_columns].fillna(data2[numerical_columns].median())


# In[42]:


data2.info()


# In[43]:


# find the object datatype -- to handle the missing value


# In[44]:


object_data=data2.select_dtypes(include=["object"])


# In[45]:


object_data.info()


# In[46]:


object_data.columns


# In[47]:


data3=data2.copy()


# In[48]:


data3[object_data.columns]=data3[object_data.columns].fillna(data3[object_data.columns].mode().iloc[0])


# In[49]:


data3.isnull().sum()


# In[50]:


sns.heatmap(data3.isnull(),cmap="coolwarm")


# In[51]:


# we have handled the misisng value


# In[52]:


# label encoding -- categorical to numerical conversion


# In[53]:


# we will apply the one hot encoding approach


# In[54]:


object_dataset=object_data.columns
object_dataset


# In[55]:


data4=pd.get_dummies(data3,columns=object_dataset,drop_first=True)


# In[56]:


data3.shape


# In[57]:


data4.shape


# In[58]:


# feature scaling we will do lateron


# In[59]:


#split the train and test dataset as given by the stake holder


# In[75]:


#train data - 1460 rows
#test data - 1459 rows


# In[61]:


training_data=data4.iloc[:1460,:]
final_test=data4.iloc[1460:,:]


# In[62]:


training_data.shape


# In[63]:


final_test.shape


# In[64]:


training_data.head()


# In[ ]:





# In[65]:


training_data=training_data.drop(["train/test_train"],axis=1)


# In[ ]:





# In[66]:


data5=data4.copy()


# In[67]:


data5.tail()


# In[68]:


data10=data5[data5["train/test_train"]==1]


# In[69]:


data10.shape


# In[70]:


data20=data5[data5["train/test_train"]==0]


# In[71]:


data20.shape


# In[ ]:





# In[ ]:


# training_data--
#final_test--


# In[72]:


training_data.head()


# In[73]:


final_test.head()


# In[74]:


final_test=final_test.drop(["SalePrice","train/test_train"],axis=1)


# In[76]:


final_test.head()


# In[77]:


final_test.shape


# In[78]:


training_data.shape


# In[79]:


training_data.head()


# In[ ]:


# feature engineering(scaling)


# In[ ]:


# training dataset


# In[85]:


X=training_data.drop(["Id","SalePrice"],axis=1)
Y=training_data["SalePrice"]


# In[82]:


from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler


# In[86]:


scaler=MinMaxScaler()
X_scaled=scaler.fit_transform(X)
X_scaled=pd.DataFrame(X_scaled,columns=X.columns)
X_scaled.head()


# In[ ]:


# scaling on testing data_set also


# In[92]:


final_test1=final_test.drop(["Id"],axis=1)


# In[93]:


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


# In[94]:


from sklearn.model_selection import train_test_split


# In[95]:


X_train,X_eval,Y_train,Y_eval=train_test_split(X_scaled,Y,test_size=0.2,random_state=100)


# In[ ]:


# lets import the algorithms to build the model


# In[96]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error


# In[97]:


lr=LinearRegression()
lr.fit(X_train,Y_train)


# In[98]:


Y_pred=lr.predict(X_eval)


# In[99]:


Y_pred_train=lr.predict(X_train)


# In[100]:


print("training_score",r2_score(Y_train,Y_pred_train))
print("evaluation_score",r2_score(Y_eval,Y_pred))


# R2=1-(RSS/TSS)
# where RSS=Sum of squared residuals
# TSS=Total sum of squares
# #### R2 score can be -ve also.

# In[ ]:


# what is Regularization /optimization.?


# In[ ]:


#Regularization technique used to solve the overfitting problem
L1 regularization-- Lasso--- penalty is the sum of absolute values of weight

L2---- ridge----penalty is the sum of square values of weight
lembda is the tuning parameter (regularization parameter)

L1/L2--- Elastic net-- hybrid behavior between L1 and L2 REGULARIZATION


# In[101]:


from sklearn.linear_model import LinearRegression,Ridge,Lasso,RidgeCV,LassoCV,ElasticNet,ElasticNetCV


# In[131]:


lasso=Lasso(alpha=50)


# In[132]:


lasso.fit(X_train,Y_train)


# In[133]:


Y_pred_lasso=lasso.predict(X_eval)


# In[134]:


Y_pred_train=lasso.predict(X_train)


# In[135]:


print("training_score",r2_score(Y_train,Y_pred_train))
print("evaluation_score",r2_score(Y_eval,Y_pred_lasso))


# In[108]:


lasso1=LassoCV(cv=10)
lasso1.fit(X_train,Y_train)
Y_pred_lasso=lasso1.predict(X_eval)
Y_pred_train=lasso1.predict(X_train)
print("training_score",r2_score(Y_train,Y_pred_train))
print("evaluation_score",r2_score(Y_eval,Y_pred_lasso))


# In[137]:


ridge=Ridge(alpha=1)
ridge.fit(X_train,Y_train)
Y_pred_ridge=ridge.predict(X_eval)
Y_pred_train=ridge.predict(X_train)
print("training_score",r2_score(Y_train,Y_pred_train))
print("evaluation_score",r2_score(Y_eval,Y_pred_ridge))


# In[110]:


ridge=RidgeCV(cv=10)
ridge.fit(X_train,Y_train)
Y_pred_ridge=ridge.predict(X_eval)
Y_pred_train=ridge.predict(X_train)
print("training_score",r2_score(Y_train,Y_pred_train))
print("evaluation_score",r2_score(Y_eval,Y_pred_ridge))


# In[113]:


en=ElasticNet()
en.fit(X_train,Y_train)
Y_pred_en=en.predict(X_eval)
Y_pred_train=en.predict(X_train)
print("training_score",r2_score(Y_train,Y_pred_train))
print("evaluation_score",r2_score(Y_eval,Y_pred_en))


# In[138]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR


# In[139]:


from sklearn.ensemble import VotingRegressor,StackingRegressor


# In[140]:


dt=DecisionTreeRegressor()
rf=RandomForestRegressor()
gdb=GradientBoostingRegressor()
ada=AdaBoostRegressor()
knn=KNeighborsRegressor()
svr=SVR()


# In[142]:


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


# In[153]:


estimators=[("dt",DecisionTreeRegressor()),
("rf",RandomForestRegressor()),
("gdb",GradientBoostingRegressor()),
("ada",AdaBoostRegressor()),
("ridge",Ridge())]


# In[154]:


voting=VotingRegressor(estimators)


# In[155]:


voting.fit(X_train,Y_train)


# In[157]:


Y_pred_vote=voting.predict(X_eval)


# In[158]:


r2_score(Y_eval,Y_pred_vote)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


#performance of GradientBoostingRegressor is better


# In[ ]:


# final testing and price prediction


# In[143]:


final_test_data.head()


# In[144]:


final_submission=final_test_data.copy()


# In[145]:


Y_final=gdb.predict(final_submission)


# In[146]:


final_submission["predicted_SP"]=Y_final


# In[147]:


final_submission.head()


# In[148]:


final_submission["ID"]=test_data["Id"]


# In[149]:


final_submission=final_submission[["ID","predicted_SP"]]


# In[150]:


final_submission


# In[ ]:





# In[ ]:




