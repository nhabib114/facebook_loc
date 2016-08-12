
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import csv as csv
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().magic(u'matplotlib inline')

from sklearn import datasets, svm, metrics


# In[2]:

in_file = 'train.csv'
locations = pd.read_csv(in_file)


# In[3]:

locations_rounded = locations.round({'x': 1, 'y': 1})
locations_sort = locations_rounded.sort_values(['x' , 'y'])
#locations_d = locations_rounded[(locations_rounded['x'] == 1.1) & (locations_rounded['y'] == 1.1)]


# In[15]:

locations_sort_slice = locations_sort[(locations_sort['x'] >= 1.0) & (locations_sort['x'] <= 2.0) &                                      (locations_sort['y'] >= 1.0) & (locations_sort['y'] <= 2.0)]

locations_sort_slice.loc[:, 'minute'] = (locations_sort_slice['time']) % 1440
locations_sort_slice.loc[:, 'hour'] = ((locations_sort_slice['time']) % 1440)/60
locations_sort_slice.loc[:, 'weekday'] = ((locations_sort_slice['time']) /(60*24)) % 7 
locations_sort_slice.loc[:, 'month'] = ((locations_sort_slice['time']) /(60*24*30)) % 12
locations_sort_slice.loc[:, 'year'] = ((locations_sort_slice['time']) /(60*24*365))
locations_sort_slice.loc[:, 'day'] = ((locations_sort_slice['time'])/(60*24)) % 365

#locations_sort_slice = locations_sort_slice[['row_id', 'x', 'y', 'accuracy', 'time', 'greater_than_800', 'place_id']]
locations_sort_slice = locations_sort_slice[['row_id', 'x', 'y', 'hour', 'weekday', 'month', 'year', 'place_id']]
locations_sort_slice = locations_sort_slice.round({'hour': 0, 'weekday': 0, 'month': 0, 'year': 0,})
#locations_sort_slice = locations_sort_slice.drop('time', 1)
locations_sort_slice


# In[16]:

from sklearn.cross_validation import train_test_split

locations_train, locations_test = train_test_split(locations_sort_slice, test_size = 0.3)

n_samples = locations_sort_slice.shape[0]
feature_cols = list(locations_train.columns[0:7])  
target_col = locations_train.columns[-1]

X_train = locations_train[feature_cols]  # feature values 
y_train = locations_train[target_col]  # corresponding targets/labels
X_test = locations_test[feature_cols]  
y_test = locations_test[target_col] 
X_train


# In[7]:

#locations_c = locations_train.round({'x': 1, 'y': 1})
#locations_d = locations_c[(locations_c['x'] == 1.1) & (locations_c['y'] == 1.1)]
#locations_time = locations_sort_slice['time'] % 1440
#locations_time
#locations_d.loc[(locations_d['time'] > 0), 'time'] = locations_d2
locations_sort_slice.groupby('place_id').count()
#test = locations_sort_slice[(locations_sort_slice['place_id'] == 1673534850)]
#locations_d.sort_values(['time'])
#test.groupby('time').count()
#test[(test['time'] < 800)]
#test.loc[:, 'greater_than_800'] = np.where(test['time'] > 800, 1, 0)
#test_count = test.groupby('greater_than_800').count()
#test_count[['row_id']].plot(kind='bar')


# In[17]:

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# In[18]:

clf = DecisionTreeClassifier(min_samples_split=100)
clf.fit(X_train, y_train)


# In[19]:

pred = clf.predict(X_test)


# In[20]:

acc = accuracy_score(pred, y_test)
print acc


# In[ ]:




# In[ ]:



