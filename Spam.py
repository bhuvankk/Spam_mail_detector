#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.model_selection import train_test_split,GridSearchCV


# In[2]:


sms = pd.read_csv('https://raw.githubusercontent.com/insaid2018/DeepLearning/master/e2e/spam.csv',encoding='ISO-8859-1')
sms.head()


# In[3]:


cols_to_drop = ['Unnamed: 2','Unnamed: 3','Unnamed: 4']
sms.drop(cols_to_drop,axis=1,inplace=True)
sms.columns = ['label','message']
sms.head()


# In[4]:


sms.isnull().sum()


# In[5]:


sms.info()


# In[6]:


cv = CountVectorizer(decode_error='ignore')
X = cv.fit_transform(sms['message'])
Y=sms['label']


# In[7]:


mnb = MultinomialNB(alpha=0.1)  # alpha set to 0.1 after checking the the GridsearchCV result


# ### Cross Validation

# In[8]:



cv_method = RepeatedStratifiedKFold(n_splits=5,  n_repeats=3, random_state=999)

cv_scores = cross_val_score(mnb, X, Y, cv=cv_method)


print(mnb, ' mean accuracy: ', round(cv_scores.mean()*100, 3), '% std: ', round(cv_scores.var()*100, 3),'%')


# In[9]:


from sklearn.model_selection import cross_validate
scores = cross_validate(mnb, X, Y, return_train_score=True)
pd.DataFrame(scores)


# In[10]:


pd.DataFrame(scores).mean()


# ### Hyperparameter tuning (Additive/ Laplacian smoothing)

# In[11]:


params = {'alpha': [0.01, 0.1, 0.5, 1.0, 10.0, ],
         }


# In[12]:


multinomial_nb_grid = GridSearchCV(MultinomialNB(), param_grid=params, n_jobs=-1,scoring='roc_auc',cv=10,return_train_score=True, verbose=5)
multinomial_nb_grid.fit(X,Y)


# In[13]:



print('Train Accuracy : %.3f'%multinomial_nb_grid.best_estimator_.score(X, Y))
#print('Test Accuracy : %.3f'%multinomial_nb_grid.best_estimator_.score(X_test, y_test))
print('Best Accuracy Through Grid Search : %.3f'%multinomial_nb_grid.best_score_)
print('Best Parameters : ',multinomial_nb_grid.best_params_)


# In[14]:


#So setting Alpha =0.1 and running again


# ### Make Predictions

# In[15]:


mnb.fit(X,Y)


# In[16]:


# just type in your message and run
your_message = 'You are the lucky winner for the lottery price of $6million.'
your_message = cv.transform([your_message])  # Transform Input to vector
claass = mnb.predict(your_message)     # Predict on Input
print(f'This is a {claass[0]} message')


# In[17]:


# just type in your message and run
your_message = 'India wins the match.'
your_message = cv.transform([your_message])
claass = mnb.predict(your_message)
print(f'This is a {claass[0]} message')


# ### Saving the model

# In[18]:


import pickle
# Creating a pickle file for the CountVectorizer
pickle.dump(cv, open('cv-transform.pkl', 'wb'))


# In[19]:


# Creating a pickle file for the Multinomial Naive Bayes model
filename = 'spam-sms-mnb-model.pkl'
pickle.dump(mnb, open(filename, 'wb'))


# In[ ]:




