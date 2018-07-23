
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import os 
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import numpy as np
from six.moves import urllib
import zipfile


# In[3]:


DATA_PATH = os.path.join('./data','battery')
DOWNLOAD_ROOT = 'https://ti.arc.nasa.gov/c/'
n = [5,9,14,15,16,17]

def fetch_battery_data(nasa_url = DOWNLOAD_ROOT, data_path = DATA_PATH):
    if not os.path.isdir(data_path):
        os.makedirs(data_path)
    n_files = []
    for f in n:
        durl = urllib.parse.urljoin(nasa_url,str(f))
        zip_path = os.path.join(data_path,"battery.zip")
        urllib.request.urlretrieve(url=durl,filename=zip_path)
        bat = zipfile.ZipFile(zip_path, 'r')
        bat.extractall(path=os.path.join(data_path,'dataset_'+str(f)))
        bat.close()
        fcnt = len([name for name in os.listdir('./data/battery/dataset_'+str(f)) if name.endswith('.mat')])
        n_files.append(fcnt)
    print("Number of .mat files downloaded: %s" % np.sum(n_files))


# In[4]:


#### uncomment to fetch data files ###
### fetch_battery_data() ###


# In[6]:


from scipy.io import loadmat
mat = loadmat(os.path.join(DATA_PATH,'dataset_5/B0005.mat'),struct_as_record=True)
print(mat.keys())


# In[7]:


matdat = mat['B0005']


# In[8]:


type(matdat)


# In[13]:


import codecs, json


# In[18]:


matlst = matdat.tolist()
#json.dump(matlst, codecs.open(r'./data/battery/B0005.json','w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)

