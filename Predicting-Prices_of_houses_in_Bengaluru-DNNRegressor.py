
# coding: utf-8

# # importing all necessary libraries.

# In[3]:


import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Loading the training dataset and separating features and reponse.

# In[4]:


prices_train = pd.read_csv("D:\intern project\machineHack\Predicting-House-Prices-In-Bengaluru-Train-Data.csv")


# In[5]:


prices_train.head()


# In[6]:


prices_train.shape


# In[7]:


prices_train.columns


# In[8]:


X = prices_train.drop('price',axis=1)


# In[9]:


y_true = prices_train['price']


# In[10]:


X.head()


# In[11]:


y_true.head()


# In[12]:


X.describe().transpose()


# # Data Cleaning

# # Total_sqft cleaned

# In[13]:


ch = ['-','S','P','A','G','C']
sam = X['total_sqft']
for i in ch:
    sam = sam.apply(lambda x: (x.split(i)))
    sam = sam.apply(lambda x: x[0])
    
X['total_sqft'] = (sam.apply(lambda x: pd.to_numeric(x)))


# ## visualizing the initial dataset.

# In[14]:


prices_train.sample(250).plot()


# # scaling the features

# In[15]:


col_to_norm = ['total_sqft', 'bath', 'balcony']


# In[16]:


X[col_to_norm] = X[col_to_norm].apply(lambda x: (x-x.min())/(x.max()-x.min()))


# In[17]:


X.head()


# In[18]:


(pd.isna(X)).sum()


# In[19]:


X["location"].fillna(" ", inplace = True)
X["size"].fillna(" ", inplace = True)
X["society"].fillna(" ", inplace = True)
X["bath"].fillna((X['bath'].mean()), inplace = True)
X["balcony"].fillna((X['balcony'].mean()), inplace = True)


# In[20]:


from sklearn.model_selection import train_test_split


# In[21]:


X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.3, random_state=18)


# In[22]:


X_test.shape


# In[23]:


area_type = tf.feature_column.categorical_column_with_hash_bucket('area_type',hash_bucket_size=10)
availability = tf.feature_column.categorical_column_with_hash_bucket('availability',hash_bucket_size=10000)
location = tf.feature_column.categorical_column_with_hash_bucket('location',hash_bucket_size=100000)
size = tf.feature_column.categorical_column_with_hash_bucket('size',hash_bucket_size=100)
society = tf.feature_column.categorical_column_with_hash_bucket('society',hash_bucket_size=100000)
total_sqft = tf.feature_column.numeric_column('total_sqft')
bath = tf.feature_column.numeric_column('bath')
balcony = tf.feature_column.numeric_column('balcony')


# In[28]:


emb_area_type = tf.feature_column.embedding_column(area_type,dimension=4)
emb_availability = tf.feature_column.embedding_column(availability,dimension=1000)
emb_location = tf.feature_column.embedding_column(location,dimension=1000)
emb_size = tf.feature_column.embedding_column(size,dimension=1000)
emb_society = tf.feature_column.embedding_column(society,dimension=1000)


# In[29]:


feat_cols = [emb_area_type,emb_availability,emb_location,emb_size,emb_society,total_sqft,bath,balcony]


# In[30]:


input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=10,num_epochs=1000,shuffle=True)


# In[31]:


model = tf.estimator.DNNRegressor(hidden_units=[8,8,8],feature_columns=feat_cols)
#model = tf.estimator.LinearRegressor(feature_columns=feat_cols)


# In[32]:


model.train(input_fn=input_func,steps = 10000)


# ## prediction to find rmse

# In[33]:


predict_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test,batch_size=10,num_epochs=1,shuffle=False)


# In[34]:


pred_gen = model.predict(predict_input_func)


# In[35]:


predictions = list(pred_gen)


# In[36]:


predictions


# In[37]:


final_pred = []

for pred in predictions:
    final_pred.append(pred['predictions'])


# In[38]:


from sklearn.metrics import mean_squared_error


# In[39]:


mean_squared_error(y_test,final_pred)**0.5


# In[40]:


prices_train.describe().transpose()


# ## Now will load the test day and will predict final values

# In[41]:


prices_test = pd.read_csv("D:\intern project\machineHack\Predicting-House-Prices-In-Bengaluru-Test-Data.csv")


# In[42]:


prices_test.head()


# In[43]:


X_eval = prices_test.drop('price',axis=1)


# ## Data Cleaning for final evalution

# In[44]:


ch = ['-','S','P','A','G','C']
sam_eval = X_eval['total_sqft']
for i in ch:
    sam_eval = sam_eval.apply(lambda x: (x.split(i)))
    sam_eval = sam_eval.apply(lambda x: x[0])
    
X_eval['total_sqft'] = (sam_eval.apply(lambda x: pd.to_numeric(x)))


# In[45]:


prices_test.sample(250).plot()


# In[46]:


X_eval[col_to_norm] = X_eval[col_to_norm].apply(lambda x: (x-x.min())/(x.max()-x.min()))


# In[47]:


X_eval.head()


# In[48]:


(pd.isna(X_eval)).sum()


# In[49]:


X_eval["location"].fillna(" ", inplace = True)
X_eval["size"].fillna(" ", inplace = True)
X_eval["society"].fillna(" ", inplace = True)
X_eval["bath"].fillna((X['bath'].mean()), inplace = True)
X_eval["balcony"].fillna((X['balcony'].mean()), inplace = True)


# In[50]:


predict_input_func_eval = tf.estimator.inputs.pandas_input_fn(x=X_eval,batch_size=10,num_epochs=1,shuffle=False)


# In[51]:


pred_gen_eval = model.predict(predict_input_func_eval)


# In[52]:


predictions_eval = list(pred_gen_eval)


# In[53]:


final_pred_eval = []

for pred in predictions_eval:
    final_pred_eval.append(pred['predictions'])


# In[54]:


prices_houses= pd.DataFrame({'price' : final_pred_eval})


# In[55]:


price_houses =[]
for i in range(len(X_eval)):
    price_houses.append(float(final_pred_eval[i]))


# In[56]:


price_houses


# In[57]:


prices_houses = pd.DataFrame({'price':price_houses})


# In[58]:


prices_houses.head()


# In[59]:


prices_houses.to_csv('D:\intern project\machineHack\House_prices_DNNRegressor.csv',index=False)

