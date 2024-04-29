#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install mlxtend


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score , precision_score ,recall_score ,f1_score ,confusion_matrix
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree


# In[3]:


# reading data
df= pd.read_csv("data.csv")


# In[4]:


df.head()


# In[5]:


for i in df.columns:
    sns.histplot(df[i],kde=True)
    plt.xticks(rotation=90)
    plt.show()


# In[ ]:





# In[6]:


# Checking for outliers
for i in df.columns[:-2]:
    sns.boxplot(df[i])
    plt.xticks(rotation=90)
    plt.show()


# ### Logistic regression

# ### Performing logistic regression method on the data to determine the accuracy in if a person is eligible for credit card or not.

# In[7]:


x_train,x_test, y_train, y_test = train_test_split(df[["age","income","creditScore","debtincomeratio"]], df["approval"], test_size=0.2, random_state=42)


# In[8]:


x_train.shape ,x_test.shape , y_train.shape , y_test.shape


# In[9]:


scaler=StandardScaler()


# In[10]:


model = LogisticRegression()


# In[11]:


model.fit(scaler.fit_transform(x_train), y_train)


# In[12]:


y_pred = model.predict(scaler.fit_transform(x_test))


# In[13]:


accuracy_score(y_test, y_pred) ,f1_score(y_test, y_pred)


# In[14]:


confusion_matrix(y_test, y_pred)


# ### Decision Tree

# ### Performing decision tree method on the data to determine the accuracy in if a person is eligible for credit card or not.

# In[15]:


df[["age","income","creditScore","debtincomeratio","approval"]].head()


# In[16]:


model = DecisionTreeClassifier()


# In[17]:


model.fit(x_train, y_train)


# In[18]:


y_pred = model.predict(x_test)


# In[19]:


accuracy_score(y_test, y_pred)


# In[ ]:





# In[20]:


plt.figure(figsize=(20,10))
plot_tree(model, feature_names=list(df["approval"].replace({1:True,0:False})), class_names=['Not Approved', 'Approved'], filled=True)
plt.show()


# In[ ]:





# ## ARM

# ### Performing ARM apriori algorithm to find the association among the items bought by customers and providing recommendations to the customer based on the item purchased.

# In[21]:


df["fbi"].head()


# In[22]:


def remove_brackets(s):
    return s.strip("[]")


# In[23]:


df['fbi'] = df['fbi'].apply(remove_brackets)


# In[24]:


df.head()


# In[ ]:





# In[25]:


data = list(df["fbi"].apply(lambda x:x.split(",") ))
data


# In[26]:


encoder = TransactionEncoder()


# In[27]:


data = encoder.fit(data).transform(data)


# In[28]:


df3 = pd.DataFrame(data,columns=encoder.columns_)


# In[29]:


df3


# In[30]:


frequent_itemsets = apriori(df3, min_support=0.01, use_colnames=True)


# In[31]:


rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
rules


# In[ ]:





# Results:
# 
# Logistic Regression: The logistic regression model achieved an accuracy score of 50% and an F1 score of 0.5 on the test data. The confusion matrix shows that the model correctly classified 2 instances of "not approved" and 2 instances of "approved," but misclassified 3 instances of "not approved" and 1 instance of "approved."
# 
# Decision Tree: The decision tree model also achieved an accuracy score of 50% on the test data similar to logistic regression model. Here, the "approval" feature is used to split the tree i.e., based on whether the application was "approved" or "not approved."
# 
# Association Rule Mining (ARM): Using the Apriori algorithm, we discovered several association rules among items bought by customers. For example, the rule ('Hardware') -> ('Air fryers') has a support of 0.025, indicating that 2.5% of transactions included both hardware and air fryers. The confidence of 1.0 suggests that if a customer purchases hardware, they are mostly guaranteed to purchase air fryers as well.
# 

# Conclusions:
# 
# 1. Credit Card Eligibility: The logistic regression and decision tree models showed limited success in predicting credit card eligibility based on age, income, credit score, and debt-to-income ratio. Both the models achieved an accuracy of only 50%, indicating that they performed no better than random guessing.
# 
# 2. Product Recommendations: The association rules generated through ARM provide valuable insights into customer purchasing behavior. By understanding which items are frequently bought together, our amazon store can optimize customer-relevant product advertising, promotions, and cross-selling strategies to increase sales and customer satisfaction.
# 
# 3. Room for Improvement: Despite the limited success of the models, there is room for improvement through feature engineering, model tuning, and incorporating additional data sources. For example, including more relevant features such as credit history or employment status may improve the predictive performance of the models. Additionally, refining the association rules by considering more transactional data or incorporating temporal patterns could enhance the accuracy of product recommendations.

# Overall, while the initial results provide some insights, further refinement and exploration are necessary to develop more robust predictive models and actionable insights for the business.

# In[ ]:




