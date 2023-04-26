#!/usr/bin/env python
# coding: utf-8

# In[14]:


#calling useful libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy import *
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from numpy import select 
from pygooglechart import PieChart3D


# In[15]:


import random
import matplotlib.colors as mcolors


# In[16]:


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# In[17]:


#reading data
df= pd.read_csv('Documents/surgical_case_durations.csv',sep =';',encoding="ISO-8859-1")
df.head()


# In[18]:



df.shape


# In[19]:


df.isnull().sum()


# In[20]:


dfFiltered = df.columns[df.isnull().sum() > 555].tolist()
df = df.drop(dfFiltered, axis=1)


# In[21]:


df = df.dropna(axis = 0)


# In[22]:


df.isnull().sum()


# In[23]:


df.head()


# In[24]:


df = df.rename(columns={'Operatietype':'OperationType', 'Chirurg':'Surgeon', 'Anesthesioloog':'Anesthesiologist', 'Casustype':'CaseType', 'Dagdeel':'DayTipe', 'Leeftijd':'Age', 'Geslacht':'Sex', 'Chronische longziekte':'Chronic lung disease', 'Extracardiale vaatpathie':'Extracardiac vascular pathology', 'Eerdere hartchirurgie':'Previous heart surgery', 'Actieve endocarditis' : 'Active Endocarditis', 'Kritische preoperatieve status':'Critical preoperative status', 'Myocard infact <90 dagen':'Myocardium infact <90 days', 'Aorta chirurgie':'Aortic surgery', 'Euroscore1':'Euroscore1', 'Slechte mobiliteit':'Poor mobility', 'Hypercholesterolemie':'Hypercholesterolemia', 'Hypertensie':'Hypertension', 'Perifeer vaatlijden':'Peripheral vascular disease', 'Geplande operatieduur':'Planned operating time', 'Operatieduur':'Operating time', 'Ziekenhuis ligduur':'Length of Stay in Hospital', 'IC ligduur':'Length of Stay in IC'})


# In[25]:


df.head()


# In[26]:


df.shape


# In[27]:


AVRdf = df[['OperationType', 'Planned operating time', 'Operating time', 'CaseType', 'DayTipe']]
AVRdf['timeDiff'] = (AVRdf['Operating time'] - AVRdf['Planned operating time'])
AVRdf.head()


# In[20]:


AVRdf['timeDiffClassified'] = np.where(abs(AVRdf['timeDiff']) < 30, 'Less Than 30 Mins', 'More Than 30 Mins')


# In[21]:


AVRdf.head()


# In[88]:


def timeRange(AVRdf):
    if (AVRdf['timeDiff'] < 0) and (AVRdf['timeDiff'] > -15):
        return '~15 mins faster than the prediction'
    elif(AVRdf['timeDiff'] <= -15) and (AVRdf['timeDiff'] > -30):
        return '~30 mins faster than the prediction'
    elif(AVRdf['timeDiff'] <= -30) and (AVRdf['timeDiff'] > -45):
        return '~45 mins faster than the prediction'
    elif (AVRdf['timeDiff'] <= -45):
        return '~1 hour faster than the prediction'
    elif(AVRdf['timeDiff'] > 0) and (AVRdf['timeDiff'] <= 15):
        return '~15 mins slower than the prediction'
    elif (AVRdf['timeDiff'] > 15) and (AVRdf['timeDiff'] <= 30):
        return '~30 mins slower than the prediction'
    elif (AVRdf['timeDiff'] > 30) and (AVRdf['timeDiff'] <= 45): 
        return '~45 mins slower than the prediction'
    elif(AVRdf['timeDiff'] > 45):
        return '~1 hour slower than the prediction'
    elif (AVRdf['timeDiff'] == 0):
        return 'on Time'

AVRdf = AVRdf.assign(timeDiffRange = AVRdf.apply(timeRange, axis =1))


# In[89]:


AVRdf


# In[ ]:





# In[90]:


AVRdf['OperationType'].value_counts().head(10)


# In[91]:


def operationType(AVRdf):
    if (AVRdf['OperationType'] == 'AVR'):
        return 'AVR'
    elif(AVRdf['OperationType'] == 'CABG'):
        return 'CABG'
    elif(AVRdf['OperationType'] == 'CABG + Pacemakerdraad tijdelijk'):
        return 'CABG + Pacemakerdraad tijdelijk'
    elif (AVRdf['OperationType'] == 'CABG + AVR'):
        return 'CABG + AVR'
    elif(AVRdf['OperationType'] == 'MVP'):
        return 'MVP'
    elif (AVRdf['OperationType'] == 'AVR + MVP shaving'):
        return 'AVR + MVP shaving'
    elif (AVRdf['OperationType'] == 'Nuss-procedure'): 
        return 'Nuss-procedure'
    elif(AVRdf['OperationType'] == 'CABG + AVR + MVP shaving'):
        return 'CABG + AVR + MVP shaving'
    elif(AVRdf['OperationType'] == 'Mediastinoscopie'):
        return 'Mediastinoscopie'
    elif(AVRdf['OperationType'] == 'CABG + Plaatsen epicardiale electrode na openen pericard + uitwendige pacemaker'):
        return 'CABG + Plaatsen epicardiale electrode na openen pericard + uitwendige pacemaker'
    else:
        return 'Others'
    
    
AVRdf = AVRdf.assign(surgeryType = AVRdf.apply(operationType, axis =1))


# In[92]:


AVRdf.head()


# In[93]:


visualizedSurgeryType1hour = AVRdf[AVRdf['timeDiffRange'] == '~1 hour faster than the prediction'].groupby('surgeryType')['surgeryType'].count() 
visualizedSurgeryType1hour = visualizedSurgeryType1hour.sort_values()
visualizedSurgeryType1hour.plot(kind='barh', figsize=(5, 5))


# In[37]:


visualizedSurgeryType45 = AVRdf[AVRdf['timeDiffRange'] == '~45 mins faster than the prediction'].groupby('surgeryType')['surgeryType'].count() 
visualizedSurgeryType45 = visualizedSurgeryType45.sort_values()
visualizedSurgeryType45.plot(kind='barh', figsize=(5, 5))


# In[38]:


visualizedSurgeryType30 = AVRdf[AVRdf['timeDiffRange'] == '~30 mins faster than the prediction'].groupby('surgeryType')['surgeryType'].count() 
visualizedSurgeryType30 = visualizedSurgeryType30.sort_values()
visualizedSurgeryType30.plot(kind='barh', figsize=(5, 5))


# In[39]:


visualizedSurgeryType15 = AVRdf[AVRdf['timeDiffRange'] == '~15 mins faster than the prediction'].groupby('surgeryType')['surgeryType'].count() 
visualizedSurgeryType15 = visualizedSurgeryType15.sort_values()
visualizedSurgeryType15.plot(kind='barh', figsize=(5, 5))


# In[41]:


visualizedSurgeryType15s = AVRdf[AVRdf['timeDiffRange'] == '~15 mins slower than the prediction'].groupby('surgeryType')['surgeryType'].count() 
visualizedSurgeryType15s = visualizedSurgeryType15s.sort_values()
visualizedSurgeryType15s.plot(kind='barh', figsize=(5, 5))


# In[42]:


visualizedSurgeryType30s = AVRdf[AVRdf['timeDiffRange'] == '~30 mins slower than the prediction'].groupby('surgeryType')['surgeryType'].count() 
visualizedSurgeryType30s = visualizedSurgeryType30s.sort_values()
visualizedSurgeryType30s.plot(kind='barh', figsize=(5, 5))


# In[43]:


visualizedSurgeryType45s = AVRdf[AVRdf['timeDiffRange'] == '~45 mins slower than the prediction'].groupby('surgeryType')['surgeryType'].count() 
visualizedSurgeryType45s = visualizedSurgeryType45s.sort_values()
visualizedSurgeryType45s.plot(kind='barh', figsize=(5, 5))


# In[45]:


visualizedSurgeryType60s = AVRdf[AVRdf['timeDiffRange'] == '~1 hour slower than the prediction'].groupby('surgeryType')['surgeryType'].count() 
visualizedSurgeryType60s = visualizedSurgeryType60s.sort_values()
visualizedSurgeryType60s.plot(kind='barh', figsize=(5, 5))


# In[46]:


visualizedSurgeryType0 = AVRdf[AVRdf['timeDiffRange'] == 'on Time'].groupby('surgeryType')['surgeryType'].count() 
visualizedSurgeryType0 = visualizedSurgeryType0.sort_values()
visualizedSurgeryType0.plot(kind='barh', figsize=(5, 5))


# In[47]:


colors = random.choices(list(mcolors.CSS4_COLORS.values()),k = 9)
def totalPercentage(x):
    print(x)
    return '{:.4f}%\n({:.0f})'.format(x, total*x/100)

visualizedSurgeryTypeCABG = AVRdf[AVRdf['surgeryType'] == 'CABG'].groupby('timeDiffRange')['timeDiffRange'].count() 
total = sum(visualizedSurgeryTypeCABG)
visualizedSurgeryTypeCABG = visualizedSurgeryTypeCABG.sort_values()
plt.pie(visualizedSurgeryTypeCABG, colors = colors, labels = visualizedSurgeryTypeCABG.index, autopct=totalPercentage, shadow = True)


# In[48]:


colors = random.choices(list(mcolors.CSS4_COLORS.values()),k = 9)
def totalPercentage(x):
    print(x)
    return '{:.4f}%\n({:.0f})'.format(x, total*x/100)

visualizedSurgeryTypeAVR = AVRdf[AVRdf['surgeryType'] == 'AVR'].groupby('timeDiffRange')['timeDiffRange'].count() 
total = sum(visualizedSurgeryTypeAVR)
visualizedSurgeryTypeAVR = visualizedSurgeryTypeAVR.sort_values()
plt.pie(visualizedSurgeryTypeAVR, colors = colors, labels = visualizedSurgeryTypeAVR.index, autopct = totalPercentage, shadow = True)


# In[50]:


colors = random.choices(list(mcolors.CSS4_COLORS.values()),k = 40)

def totalPercentage(x):
    print(x)
    return '{:.4f}%\n({:.0f})'.format(x, total*x/100)

visualizedSurgeryTypeCP = AVRdf[AVRdf['surgeryType'] == 'CABG + Pacemakerdraad tijdelijk'].groupby('timeDiffRange')['timeDiffRange'].count() 
total = sum(visualizedSurgeryTypeCP)
visualizedSurgeryTypeCP = visualizedSurgeryTypeCP.sort_values()
plt.pie(visualizedSurgeryTypeCP, colors = colors, labels = visualizedSurgeryTypeCP.index, autopct = totalPercentage, shadow = True)


# In[51]:


colors = random.choices(list(mcolors.CSS4_COLORS.values()),k = 9)
def totalPercentage(x):
    print(x)
    return '{:.4f}%\n({:.0f})'.format(x, total*x/100)

visualizedSurgeryTypeCA = AVRdf[AVRdf['surgeryType'] == 'CABG + AVR'].groupby('timeDiffRange')['timeDiffRange'].count() 
total = sum(visualizedSurgeryTypeCA)
visualizedSurgeryTypeCA = visualizedSurgeryTypeCA.sort_values()
plt.pie(visualizedSurgeryTypeCA, colors = colors, labels = visualizedSurgeryTypeCA.index, autopct = totalPercentage, shadow = True)


# In[52]:


colors = random.choices(list(mcolors.CSS4_COLORS.values()),k = 30)

def totalPercentage(x):
    print(x)
    return '{:.4f}%\n({:.0f})'.format(x, total*x/100)

visualizedSurgeryTypeMVP = AVRdf[AVRdf['surgeryType'] == 'MVP'].groupby('timeDiffRange')['timeDiffRange'].count() 
total = sum(visualizedSurgeryTypeMVP)
visualizedSurgeryTypeMVP = visualizedSurgeryTypeMVP.sort_values()
plt.pie(visualizedSurgeryTypeMVP, colors = colors, labels = visualizedSurgeryTypeMVP.index, autopct = totalPercentage, shadow = True)


# In[54]:


colors = random.choices(list(mcolors.CSS4_COLORS.values()),k = 40)

def totalPercentage(x):
    print(x)
    return '{:.4f}%\n({:.0f})'.format(x, total*x/100)

visualizedSurgeryTypeAMs = AVRdf[AVRdf['surgeryType'] == 'AVR + MVP shaving'].groupby('timeDiffRange')['timeDiffRange'].count() 
total = sum(visualizedSurgeryTypeAMs)
visualizedSurgeryTypeAMs = visualizedSurgeryTypeAMs.sort_values()
plt.pie(visualizedSurgeryTypeAMs, colors = colors, labels = visualizedSurgeryTypeAMs.index, autopct = totalPercentage, shadow = True)


# In[55]:


colors = random.choices(list(mcolors.CSS4_COLORS.values()),k = 40)

def totalPercentage(x):
    print(x)
    return '{:.4f}%\n({:.0f})'.format(x, total*x/100)

visualizedSurgeryTypeNP = AVRdf[AVRdf['surgeryType'] == 'Nuss-procedure'].groupby('timeDiffRange')['timeDiffRange'].count() 
total = sum(visualizedSurgeryTypeNP)
visualizedSurgeryTypeNP = visualizedSurgeryTypeNP.sort_values()
plt.pie(visualizedSurgeryTypeNP, colors = colors, labels = visualizedSurgeryTypeNP.index, autopct = totalPercentage, shadow = True)


# In[95]:


colors = random.choices(list(mcolors.CSS4_COLORS.values()),k = 40)

def totalPercentage(x):
    print(x)
    return '{:.4f}%\n({:.0f})'.format(x, total*x/100)

visualizedSurgeryTypeCAM = AVRdf[AVRdf['surgeryType'] == 'CABG + AVR + MVP shaving'].groupby('timeDiffRange')['timeDiffRange'].count() 
total = sum(visualizedSurgeryTypeCAM)
visualizedSurgeryTypeCAM = visualizedSurgeryTypeCAM.sort_values()
plt.pie(visualizedSurgeryTypeCAM, colors = colors, labels = visualizedSurgeryTypeCAM.index, autopct = totalPercentage, shadow = True)


# In[96]:


colors = random.choices(list(mcolors.CSS4_COLORS.values()),k = 40)

def totalPercentage(x):
    print(x)
    return '{:.4f}%\n({:.0f})'.format(x, total*x/100)

visualizedSurgeryTypeM = AVRdf[AVRdf['surgeryType'] == 'Mediastinoscopie'].groupby('timeDiffRange')['timeDiffRange'].count() 
total = sum(visualizedSurgeryTypeM)
visualizedSurgeryTypeM = visualizedSurgeryTypeM.sort_values()
plt.pie(visualizedSurgeryTypeM, colors = colors, labels = visualizedSurgeryTypeM.index, autopct = totalPercentage, shadow = True)


# In[97]:


colors = random.choices(list(mcolors.CSS4_COLORS.values()),k = 40)

def totalPercentage(x):
    print(x)
    return '{:.4f}%\n({:.0f})'.format(x, total*x/100)

visualizedSurgeryTypeCPU = AVRdf[AVRdf['surgeryType'] == 'CABG + Plaatsen epicardiale electrode na openen pericard + uitwendige pacemaker'].groupby('timeDiffRange')['timeDiffRange'].count() 
total = sum(visualizedSurgeryTypeCPU)
visualizedSurgeryTypeCPU = visualizedSurgeryTypeCPU.sort_values()
plt.pie(visualizedSurgeryTypeCPU, colors = colors, labels = visualizedSurgeryTypeCPU.index, autopct = totalPercentage, shadow = True)


# In[63]:


colors = random.choices(list(mcolors.CSS4_COLORS.values()),k = 40)

def totalPercentage(x):
    print(x)
    return '{:.4f}%\n({:.0f})'.format(x, total*x/100)

visualizedSurgeryTypeO = AVRdf[AVRdf['surgeryType'] == 'Others'].groupby('timeDiffRange')['timeDiffRange'].count() 
total = sum(visualizedSurgeryTypeO)
visualizedSurgeryTypeO = visualizedSurgeryTypeO.sort_values()
plt.pie(visualizedSurgeryTypeO, colors = colors, labels = visualizedSurgeryTypeO.index, autopct = totalPercentage, shadow = True)


# In[64]:


visualizedSurgeryTypeDT = AVRdf[AVRdf['timeDiffRange'] != 'onTime'].groupby('DayTipe')['DayTipe'].count() 
visualizedSurgeryTypeDT = visualizedSurgeryTypeDT.sort_values()
visualizedSurgeryTypeDT.plot(kind='pie', figsize=(10, 40))


# In[65]:


visualizedSurgeryTypeDT = AVRdf[AVRdf['timeDiffRange'] != 'onTime'].groupby('CaseType')['CaseType'].count() 
visualizedSurgeryTypeDT = visualizedSurgeryTypeDT.sort_values()
visualizedSurgeryTypeDT.plot(kind='pie', figsize=(10, 40))


# In[66]:


visualizedSurgeryTypeDT = AVRdf[AVRdf['CaseType'] == 'Spoed'].groupby('timeDiffRange')['timeDiffRange'].count() 
visualizedSurgeryTypeDT = visualizedSurgeryTypeDT.sort_values()
visualizedSurgeryTypeDT.plot(kind='barh', figsize=(10, 5))


# In[ ]:




