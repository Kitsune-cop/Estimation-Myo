#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import numpy as np
import os


# In[12]:


path = './data-00/'
x = []
y = []
count = 0


# In[10]:


# True in [str(i) in '41.csv' for i in range(40,61)]


# In[13]:


for p_name in os.listdir(os.path.join(path)):
    for t_name in os.listdir(os.path.join(path,p_name)):
        for file_name in os.listdir(os.path.join(path,p_name,t_name)):
            # if 'emg.csv' in file_name and '_r_' in file_name:
            # if not (True in [str(i) in  file_name for i in range(40,61)]):
                df = pd.read_csv(os.path.join(path, p_name, t_name ,file_name), header=None)
                if len(df) >= 200:
                    print(file_name, len(df))
                    x.append(df.iloc[0:200, :])
                    # x.append(df.iloc[:lenge, 2:].transpose())
                    if 'paper' in file_name:
                        y.append([2])
                    elif 'rock' in file_name:
                        y.append([1])
                    elif 'scissor' in file_name:
                        y.append([3])
                    elif 'relax' in file_name:
                        y.append([0])
                    count += 1


# In[14]:


X = np.array(x)
Y = np.array(y)


# In[15]:


# X = X/127
X.shape, Y.shape
# X[0][1]


# In[16]:


np.savetxt("./attribute_200_8.csv", X.reshape((-1, X.shape[-1])), delimiter=',')
np.savetxt("./target_200_8.csv", Y.reshape((-1, Y.shape[-1])), delimiter=',')


# In[ ]:




