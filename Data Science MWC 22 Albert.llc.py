#!/usr/bin/env python
# coding: utf-8

# ## Objetivos: 
# Haz un analisis exploratorio de los datos que permita:
# 1. Analizar las ventas y la actividad de los clientes
# 2. Evaluar el impacto de la promoción
# 

# ## Import library

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



pd.options.display.float_format = '{:.2f}'.format #with out exponential
pd.set_option('display.max_columns', None) #Open full columns


# ## Import Datasets

# In[2]:


#data_info=webbrowser.open('https://nuwe.io/challenge/mwc-22-data')
url_client_tab='https://challenges-asset-files.s3.us-east-2.amazonaws.com/data_sets/Data-Science/4+-+events/mwc22/mwc22-client_table.csv'
client_tab=pd.read_csv(url_client_tab,decimal=',')
url_orders_tab='https://challenges-asset-files.s3.us-east-2.amazonaws.com/data_sets/Data-Science/4+-+events/mwc22/mwc22-orders_table.csv'
orders_tab=pd.read_csv(url_orders_tab)


# In[3]:


url_test='https://challenges-asset-files.s3.us-east-2.amazonaws.com/data_sets/Data-Science/4+-+events/mwc22/mwc22-client_table+-+test_x.csv'
dftest=pd.read_csv(url_test,decimal=',')


# ## Data Cleaning

# ### Orders table

# In[4]:


orders_tab.describe()


# In[5]:


orders_tab


# In[6]:


# Null values
orders_tab[orders_tab.isnull().any(axis=1)]


# In[7]:


dupl_orders_tab=orders_tab[orders_tab.duplicated(keep=False)]
dupl_orders_tab=dupl_orders_tab.sort_values(by='ORDER ID',ascending=False)
dupl_orders_tab


# In[8]:


#Drop duplicates
orders_tab.drop_duplicates(keep='first', inplace=True)


# In[9]:


print(len(orders_tab['FRUIT_PRODUCT'].unique()))
orders_tab['FRUIT_PRODUCT'].unique()


# In[10]:


plt.title('Fruit Product Histogram')
plt.hist(orders_tab['FRUIT_PRODUCT'], bins = 24)
plt.xticks(rotation=90)
plt.grid(True)
plt.show()
plt.clf()


# In[11]:


orders_byNuwe_Fruit=orders_tab[orders_tab['FRUIT_PRODUCT']=='Nuwe Fruit']
orders_byNuwe_Fruit.sort_values(by='CLIENT ID',ascending=False)


# In[12]:


orders_byNuwe_Fruit=orders_tab[orders_tab['ORDER ID']==668265264]
orders_byNuwe_Fruit.sort_values(by='CLIENT ID',ascending=False)


# In[13]:


orders_negative=orders_tab[orders_tab['NB PRODS']<1]
orders_negative.sort_values(by='ORDER ID',ascending=False)


# In[14]:


negativeOrders=orders_negative.sort_values(by='ORDER ID',ascending=False)['ORDER ID']
negativeOrders


# In[15]:


negativeOrders.unique()


# In[16]:


difOrders=orders_tab[orders_tab['ORDER ID'].isin(negativeOrders)].groupby(['ORDER ID']).sum()['NB PRODS'].sort_values()
#difOrders=pd.DataFrame(data=difOrders)
difOrders
#returnsFullOrders=difOrders[difOrders['NB PRODS']==0]
#returnsFullOrders=returnsFullOrders.reset_index()
#returnsFullOrders


# In[17]:


difOrders


# In[18]:


orders_byNuwe_Fruit['NB PRODS'].unique()


# ### Client table

# In[19]:


client_tab


# In[20]:


client_tab.describe()


# In[21]:


# Null values
client_tab[client_tab.isnull().any(axis=1)]


# In[22]:


dupli_client_tab=client_tab[client_tab.duplicated(subset=['CLIENT ID'],keep=False)]
dupli_client_tab=dupli_client_tab.sort_values(by='CLIENT ID',ascending=False)
dupli_client_tab


# In[23]:


client_tab


# In[24]:


plt.hist(client_tab['CLIENT_SEGMENT'])
plt.title('Count of each segment')
plt.ylabel('Count')
plt.xlabel('Segments')
plt.show()
plt.hist(client_tab['RECEIVED_COMMUNICATION'])
plt.title('Count promotion and not promotion')
plt.ylabel('Count')
plt.xlabel('Received communication')
plt.show()


# In[25]:


plt.scatter(x=client_tab['AVG CONSO'],
            y=client_tab['AVG BASKET SIZE'],
            c=client_tab['CLIENT_SEGMENT'],
           )
plt.title('Average consum with average basket size for each segment')
plt.ylabel('Average basket size')
plt.xlabel('Average consume')
plt.show()


# ### Data cleaning client table and orders table

# In[26]:


df=pd.merge(client_tab,orders_tab, on='CLIENT ID')
df


# In[27]:


df=df.rename(columns={"CLIENT ID": "CLIENT_ID", 
                   "AVG CONSO": "AVG_CONSO",
                   "AVG BASKET SIZE":"AVG_BASKET_SIZE",
                  "NB PRODS":"NB_PRODS",
                  "ORDER ID":"ORDER_ID"})


# In[28]:


df


# In[29]:


df['prod_fruit']=df['FRUIT_PRODUCT']
df1=pd.get_dummies(df, columns=["prod_fruit"],prefix="",prefix_sep="")
df1=df1.sort_values(by=['ORDER_ID'])
df=df.rename(columns={"Nuwe Fruit": "Nuwe_Fruit", 
                   "Devil Fruit": "Devil_Fruit",
                   })
for i in range(len(df1)):
    fruit=df1.loc[i,'FRUIT_PRODUCT']
    num_prod=df1.loc[i,'NB_PRODS']
    df1.loc[i,fruit]=num_prod


# In[30]:


#Delet columns to use groupby
df1=df1.drop(['FRUIT_PRODUCT','NB_PRODS'],axis=1)
#Columns need Group
var_client=list(df1.columns[:6])
#Columns need to sum
var_fruit=list(df1.columns[-24:-1])
#Group by
df1_aux=df1.groupby(var_client)[var_fruit].apply(lambda x : x.astype(int).sum()).reset_index()


# ## EDA

# ### Client

# #### Productos mas comprados por los clientes

# In[31]:


#Productos mas comprados por los clientes
total_fruit= df1.iloc[:, -24:-1].sum(axis=0)
total_fruit=total_fruit.reset_index()
total_fruit=total_fruit.rename(columns={"index": "type_fruit", 0: "sum",})
total_fruit=total_fruit.sort_values(by=['sum'],ascending=False)

#Plot
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.barh(total_fruit['type_fruit'],total_fruit['sum'])
plt.title('Productos más vendidos*,\n *un producto equivale a 10 piezas de fruta')
plt.ylabel('Fruta')
plt.xlabel('Cantidad de productos comprados')
plt.show()


# #### How many clients has Nuwe Fruit

# In[32]:


#Number of Client
len(df1['CLIENT_ID'].unique())


# #### Los clientes mas fieles

# In[33]:


# CLIENT_ID as a string
df1['CLIENT_ID'] = df1['CLIENT_ID'].apply(str)
col_fruit=list(df1.columns[-24:-1])
orders_byClient_bought=df1.groupby(['CLIENT_ID','ORDER_ID'])[col_fruit].apply(lambda x : x.astype(int).sum()).reset_index()
# Most current customers
client_satisf=orders_byClient_bought.groupby(['CLIENT_ID'])['ORDER_ID'].count().reset_index()
client_satisf=client_satisf.sort_values(by='ORDER_ID',ascending=False)
client_satisf


# In[34]:


#Top 5 client
top5client=client_satisf.sort_values(by='ORDER_ID',ascending=False).iloc[:5,:]
top5client


# <span class="burk">#### Client Segment</span>

# In[35]:


# How many clients are in each segment:
segment_client=df1.groupby(['CLIENT_SEGMENT'])['CLIENT_ID'].count().reset_index()
segment_client
# Plot of customers by segment
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(segment_client['CLIENT_SEGMENT'],segment_client['CLIENT_ID'])
plt.title('Customers in each segment')
plt.ylabel('Amount of customers')
plt.xlabel('Segments')
plt.show()


# ## Predict segment 

# In[36]:


client_tab.drop(columns=['CLIENT ID'])


# In[ ]:





# ### Simple model

# In[37]:


client_tab['CLIENT_SEGMENT'] = client_tab['CLIENT_SEGMENT'].apply(str)


# In[38]:


client_tab_feat=['AVG CONSO','AVG BASKET SIZE','RECEIVED_COMMUNICATION']
y=client_tab.loc[:,'CLIENT_SEGMENT']
x=client_tab.loc[:,client_tab_feat]


# In[39]:


import random
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report,cohen_kappa_score,confusion_matrix
from sklearn import svm
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingClassifier


# In[40]:


#Split in train and test for x and y
random.seed(101)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7)


# In[41]:


KNN_simple = KNeighborsClassifier(n_neighbors=10).fit(x_train, y_train)
print(cross_val_score(KNN_simple, x_test, y_test, cv=5))

#Predictions
KNN_simple_pred = KNN_simple.predict(x_test)

#Performance Metrics (Errors)
print("Accuracy:", accuracy_score(y_test, KNN_simple_pred))
print("Kappa:", cohen_kappa_score(y_test, KNN_simple_pred))
print(classification_report(y_test, KNN_simple_pred))
confusion_matrix(y_test, KNN_simple_pred)


# In[42]:


SVC_simple = svm.SVC().fit(x_train, y_train)
print(cross_val_score(SVC_simple, x_test, y_test, cv=5))

#Predictions
SVC_simple_pred = SVC_simple.predict(x_test)

#Performance Metrics (Errors)
print("Accuracy:", accuracy_score(y_test, SVC_simple_pred))
print("Kappa:", cohen_kappa_score(y_test, SVC_simple_pred))
print(classification_report(y_test, SVC_simple_pred))
confusion_matrix(y_test, SVC_simple_pred)


# In[ ]:


GBC_simple = GradientBoostingClassifier().fit(x_train ,y_train)
print(cross_val_score(GBC_simple, x_test, y_test, cv=5))

#Predictions
GBC_simple_pred = GBC_simple.predict(x_test)

#Performance Metrics (Errors)
print("Accuracy:", accuracy_score(y_test, GBC_simple_pred))
print("Kappa:", cohen_kappa_score(y_test, GBC_simple_pred))
print(classification_report(y_test, GBC_simple_pred))
confusion_matrix(y_test, GBC_simple_pred)


# In[ ]:


result_simple = [["Model 1: GBC", accuracy_score(y_test, GBC_simple_pred),
                cohen_kappa_score(y_test, GBC_simple_pred)],
                ["Model 2: KNN", accuracy_score(y_test, KNN_simple_pred),
                cohen_kappa_score(y_test, KNN_simple_pred)],
                  ["Model 3: SVC", accuracy_score(y_test, SVC_simple_pred),
                cohen_kappa_score(y_test, SVC_simple_pred)]]

print('Gradient Boosting Classifier the best model to predict Segment with a basic model.')
data_simple_result = pd.DataFrame(result_simple, columns = ['Model', 'Accuracy','Kappa'])
data_simple_result


# In[ ]:


### Prediction from hold out


# In[ ]:


dftest


# In[ ]:


dftest_feat=['AVG CONSO','AVG BASKET SIZE','RECEIVED_COMMUNICATION']
x=dftest.loc[:,dftest_feat]


# In[ ]:


#Prediction:
GBC_simple_pred_test = GBC_simple.predict(x)


# In[ ]:


dftest['CLIENT_SEGMENT']=GBC_simple_pred_test


# In[ ]:


dftest=dftest.drop(columns=['AVG CONSO','AVG BASKET SIZE','RECEIVED_COMMUNICATION'])


# In[ ]:


dftest


# In[ ]:


# Create csv
dftest.to_csv('/Users/albertlloveras/Data analytics - Ubiqum/Datathone/results.csv')  

