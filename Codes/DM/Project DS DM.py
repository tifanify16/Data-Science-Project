#!/usr/bin/env python
# coding: utf-8

# In[1]:


# from multiprocessing.context import ForkServerContext
# from xml.sax.handler import feature_external_ges
import pandas as pd
import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


# In[2]:


# import and clearning data
df = pd.read_csv('Documents/surgical_case_durations.csv', sep=";", encoding='ISO-8859-1')
df.isnull().sum()
dfFiltered = df.columns[df.isnull().sum() > 555].tolist()
df = df.drop(dfFiltered, axis=1)
df.isnull().sum()
df = df.dropna(axis = 0)
df = df.rename(columns={'Operatietype':'OperationType', 'Chirurg':'Surgeon', 'Anesthesioloog':'Anesthesiologist', 'Casustype':'CaseType', 'Dagdeel':'DayTipe', 'Leeftijd':'Age', 'Geslacht':'Sex', 'Chronische longziekte':'Chroniclungdisease', 'Extracardiale vaatpathie':'Extracardiacvascularpathology', 'Eerdere hartchirurgie':'Previousheartsurgery', 'Actieve endocarditis' : 'ActiveEndocarditis', 'Kritische preoperatieve status':'Criticalpreoperativestatus', 'Myocard infact <90 dagen':'Myocardiuminfact', 'Aorta chirurgie':'Aorticsurgery', 'Euroscore1':'Euroscore1', 'Slechte mobiliteit':'Poormobility', 'Hypercholesterolemie':'Hypercholesterolemia', 'Hypertensie':'Hypertension', 'Perifeer vaatlijden':'Peripheralvasculardisease', 'Geplande operatieduur':'Plannedoperatingtime', 'Operatieduur':'Operatingtime', 'Ziekenhuis ligduur':'LengthofStayinHospital', 'IC ligduur':'LengthofStayinIC'})


# In[3]:


pip install xgboost


# In[4]:


# Descriptive statistics and correlation analysis

# Filter unimportant features using random forest algorithm
variable_cols = ['OperationType','Surgeon','Anesthesiologist','CaseType','DayTipe','Age','Sex','Chroniclungdisease','Extracardiacvascularpathology','Previousheartsurgery','ActiveEndocarditis', 'Criticalpreoperativestatus','Myocardiuminfact','Aorticsurgery','Euroscore1','Poormobility','Hypercholesterolemia','Hypertension','Peripheralvasculardisease','LengthofStayinHospital','LengthofStayinIC']
encoder_OperationType = LabelEncoder().fit(df["OperationType"])
df["OperationType"] = encoder_OperationType.transform(df["OperationType"])
encoder_Surgeon = LabelEncoder().fit(df["Surgeon"])
df["Surgeon"] = encoder_Surgeon.transform(df["Surgeon"])
encoder_Anesthesiologist = LabelEncoder().fit(df["Anesthesiologist"])
df["Anesthesiologist"] = encoder_Anesthesiologist.transform(df["Anesthesiologist"])
encoder_CaseType = LabelEncoder().fit(df["CaseType"])
df["CaseType"] = encoder_CaseType.transform(df["CaseType"])
encoder_DayTipe = LabelEncoder().fit(df["DayTipe"])
df["DayTipe"] = encoder_DayTipe.transform(df["DayTipe"])
encoder_Age = LabelEncoder().fit(df["Age"])
df["Age"] = encoder_Age.transform(df["Age"])
encoder_Sex = LabelEncoder().fit(df["Sex"])
df["Sex"] = encoder_Sex.transform(df["Sex"])
encoder_Chroniclungdisease = LabelEncoder().fit(df["Chroniclungdisease"])
df["Chroniclungdisease"] = encoder_Chroniclungdisease.transform(df["Chroniclungdisease"])
encoder_Extracardiacvascularpathology = LabelEncoder().fit(df["Extracardiacvascularpathology"])
df["Extracardiacvascularpathology"] = encoder_Extracardiacvascularpathology.transform(df["Extracardiacvascularpathology"])
encoder_Previousheartsurgery = LabelEncoder().fit(df["Previousheartsurgery"])
df["Previousheartsurgery"] = encoder_Previousheartsurgery.transform(df["Previousheartsurgery"])
encoder_ActiveEndocarditis = LabelEncoder().fit(df["ActiveEndocarditis"])
df["ActiveEndocarditis"] = encoder_ActiveEndocarditis.transform(df["ActiveEndocarditis"])
encoder_Criticalpreoperativestatus = LabelEncoder().fit(df["Criticalpreoperativestatus"])
df["Criticalpreoperativestatus"] = encoder_Criticalpreoperativestatus.transform(df["Criticalpreoperativestatus"])
encoder_Myocardiuminfact = LabelEncoder().fit(df["Myocardiuminfact"])
df["Myocardiuminfact"] = encoder_Myocardiuminfact.transform(df["Myocardiuminfact"])
encoder_Aorticsurgery = LabelEncoder().fit(df["Aorticsurgery"])
df["Aorticsurgery"] = encoder_Aorticsurgery.transform(df["Aorticsurgery"])
encoder_Euroscore1 = LabelEncoder().fit(df["Euroscore1"])
df["Euroscore1"] = encoder_Euroscore1.transform(df["Euroscore1"])
encoder_Poormobility= LabelEncoder().fit(df["Poormobility"])
df["Poormobility"] = encoder_Poormobility.transform(df["Poormobility"])
encoder_Hypercholesterolemia = LabelEncoder().fit(df["Hypercholesterolemia"])
df["Hypercholesterolemia"] = encoder_Hypercholesterolemia.transform(df["Hypercholesterolemia"])
encoder_Hypertension = LabelEncoder().fit(df["Hypertension"])
df["Hypertension"] = encoder_Hypertension.transform(df["Hypertension"])
encoder_Peripheralvasculardisease = LabelEncoder().fit(df["Peripheralvasculardisease"])
df["Peripheralvasculardisease"] = encoder_Peripheralvasculardisease.transform(df["Peripheralvasculardisease"])
encoder_LengthofStayinHospital = LabelEncoder().fit(df["LengthofStayinHospital"])
df["LengthofStayinHospital"] = encoder_LengthofStayinHospital.transform(df["LengthofStayinHospital"])
encoder_LengthofStayinIC = LabelEncoder().fit(df["LengthofStayinIC"])
df["LengthofStayinIC"] = encoder_LengthofStayinIC.transform(df["LengthofStayinIC"])


# In[5]:


x = df[variable_cols]
y = df["Operatingtime"]
x_train, x_test, y_train, y_test  = train_test_split(x, y, test_size=0.4)
# Random Forest Classifier model with default parameter
# rfc = RandomForestClassifier(random_state=0)
# rfc.fit(x_train,y_train)
# y_pred = rfc.predict(x_test)
# print('10 decision-trees : {0:0.8f}'. format(accuracy_score(y_test, y_pred)))
# Random Forest Classifier model with 1000 parameter
rfc_1000 = RandomForestClassifier(n_estimators=1000,random_state=0)
rfc_1000.fit(x_train,y_train)
y_pred_1000 = rfc_1000.predict(x_test)
# score2 = accuracy_score(y_test, y_pred_1000)
# print('1000 decision-trees : {0:0.8f}'. format(accuracy_score(y_test, y_pred_1000)))
# view feature scores
feature_scores = pd.Series(rfc_1000.feature_importances_, index=x_train.columns).sort_values(ascending=False)


# In[6]:


visualized = feature_scores.sort_values()


# In[7]:


visualized.plot(kind = 'barh', figsize = (10,5))


# In[28]:


import pandas as pd
import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Import data
df = pd.read_csv('Documents/surgical_case_durations.csv', sep=";", encoding='ISO-8859-1')
df.isnull().sum()
dfFiltered = df.columns[df.isnull().sum() > 555].tolist()
df = df.drop(dfFiltered, axis=1)
df.isnull().sum()
df = df.dropna(axis = 0)
df = df.rename(columns={'Operatietype':'OperationType', 'Chirurg':'Surgeon', 'Anesthesioloog':'Anesthesiologist', 'Casustype':'CaseType', 'Dagdeel':'DayTipe', 'Leeftijd':'Age', 'Geslacht':'Sex', 'Chronische longziekte':'Chroniclungdisease', 'Extracardiale vaatpathie':'Extracardiacvascularpathology', 'Eerdere hartchirurgie':'Previousheartsurgery', 'Actieve endocarditis' : 'ActiveEndocarditis', 'Kritische preoperatieve status':'Criticalpreoperativestatus', 'Myocard infact <90 dagen':'Myocardiuminfact', 'Aorta chirurgie':'Aorticsurgery', 'Euroscore1':'Euroscore1', 'Slechte mobiliteit':'Poormobility', 'Hypercholesterolemie':'Hypercholesterolemia', 'Hypertensie':'Hypertension', 'Perifeer vaatlijden':'Peripheralvasculardisease', 'Geplande operatieduur':'Plannedoperatingtime', 'Operatieduur':'Operatingtime', 'Ziekenhuis ligduur':'LengthofStayinHospital', 'IC ligduur':'LengthofStayinIC'})

def OperationType(df):
    if (df['OperationType'] == 'AVR'):
        return 'AVR'
    elif(df['OperationType'] == 'CABG'):
        return 'CABG'
    elif(df['OperationType'] == 'CABG + Pacemakerdraad tijdelijk'):
        return 'CABG + Pacemakerdraad tijdelijk'
    elif(df['OperationType'] == 'CABG + AVR'):
        return 'CABG + AVR'
    elif(df['OperationType'] == 'MVP'):
        return 'MVP'
    elif(df['OperationType'] == 'AVR + MVP shaving'):
        return 'AVR + MVP shaving'
    elif(df['OperationType'] == 'VATS Boxlaesie'): 
        return 'VATS Boxlaesie'
    elif(df['OperationType'] == 'CABG + AVR + MVP shaving'):
        return 'CABG + AVR + MVP shaving'
    elif(df['OperationType'] == 'Bentall procedure'):
        return 'Bentall procedure'
    elif(df['OperationType'] == 'CABG + Plaatsen epicardiale electrode na openen pericard + uitwendige pacemaker'):
        return 'CABG + Plaatsen epicardiale electrode na openen pericard + uitwendige pacemaker'
    else:
        return 'Others'
    
    
df = df.assign(SurgeryType = df.apply(OperationType, axis =1))

# Discretization
bin_labels_3 =["low","medium","high"]
df["time"] = pd.qcut(df["Operatingtime"], q = [0, .33, .66, 1], labels = bin_labels_3)
print(df.describe)

# Feature selection

variable_cols = ['SurgeryType','Surgeon','Anesthesiologist','CaseType','DayTipe','Age','Sex','Chroniclungdisease','Extracardiacvascularpathology','Previousheartsurgery','ActiveEndocarditis', 'Criticalpreoperativestatus','Myocardiuminfact','Aorticsurgery','Euroscore1','Poormobility','Hypercholesterolemia','Hypertension','Peripheralvasculardisease','LengthofStayinHospital','LengthofStayinIC']
encoder_SurgeryType = LabelEncoder().fit(df["SurgeryType"])
df["SurgeryType"] = encoder_SurgeryType.transform(df["SurgeryType"])
encoder_Surgeon = LabelEncoder().fit(df["Surgeon"])
df["Surgeon"] = encoder_Surgeon.transform(df["Surgeon"])
encoder_Anesthesiologist = LabelEncoder().fit(df["Anesthesiologist"])
df["Anesthesiologist"] = encoder_Anesthesiologist.transform(df["Anesthesiologist"])
encoder_CaseType = LabelEncoder().fit(df["CaseType"])
df["CaseType"] = encoder_CaseType.transform(df["CaseType"])
encoder_DayTipe = LabelEncoder().fit(df["DayTipe"])
df["DayTipe"] = encoder_DayTipe.transform(df["DayTipe"])
encoder_Age = LabelEncoder().fit(df["Age"])
df["Age"] = encoder_Age.transform(df["Age"])
encoder_Sex = LabelEncoder().fit(df["Sex"])
df["Sex"] = encoder_Sex.transform(df["Sex"])
encoder_Chroniclungdisease = LabelEncoder().fit(df["Chroniclungdisease"])
df["Chroniclungdisease"] = encoder_Chroniclungdisease.transform(df["Chroniclungdisease"])
encoder_Extracardiacvascularpathology = LabelEncoder().fit(df["Extracardiacvascularpathology"])
df["Extracardiacvascularpathology"] = encoder_Extracardiacvascularpathology.transform(df["Extracardiacvascularpathology"])
encoder_Previousheartsurgery = LabelEncoder().fit(df["Previousheartsurgery"])
df["Previousheartsurgery"] = encoder_Previousheartsurgery.transform(df["Previousheartsurgery"])
encoder_ActiveEndocarditis = LabelEncoder().fit(df["ActiveEndocarditis"])
df["ActiveEndocarditis"] = encoder_ActiveEndocarditis.transform(df["ActiveEndocarditis"])
encoder_Criticalpreoperativestatus = LabelEncoder().fit(df["Criticalpreoperativestatus"])
df["Criticalpreoperativestatus"] = encoder_Criticalpreoperativestatus.transform(df["Criticalpreoperativestatus"])
encoder_Myocardiuminfact = LabelEncoder().fit(df["Myocardiuminfact"])
df["Myocardiuminfact"] = encoder_Myocardiuminfact.transform(df["Myocardiuminfact"])
encoder_Aorticsurgery = LabelEncoder().fit(df["Aorticsurgery"])
df["Aorticsurgery"] = encoder_Aorticsurgery.transform(df["Aorticsurgery"])
encoder_Euroscore1 = LabelEncoder().fit(df["Euroscore1"])
df["Euroscore1"] = encoder_Euroscore1.transform(df["Euroscore1"])
encoder_Poormobility= LabelEncoder().fit(df["Poormobility"])
df["Poormobility"] = encoder_Poormobility.transform(df["Poormobility"])
encoder_Hypercholesterolemia = LabelEncoder().fit(df["Hypercholesterolemia"])
df["Hypercholesterolemia"] = encoder_Hypercholesterolemia.transform(df["Hypercholesterolemia"])
encoder_Hypertension = LabelEncoder().fit(df["Hypertension"])
df["Hypertension"] = encoder_Hypertension.transform(df["Hypertension"])
encoder_Peripheralvasculardisease = LabelEncoder().fit(df["Peripheralvasculardisease"])
df["Peripheralvasculardisease"] = encoder_Peripheralvasculardisease.transform(df["Peripheralvasculardisease"])
encoder_LengthofStayinHospital = LabelEncoder().fit(df["LengthofStayinHospital"])
df["LengthofStayinHospital"] = encoder_LengthofStayinHospital.transform(df["LengthofStayinHospital"])
encoder_LengthofStayinIC = LabelEncoder().fit(df["LengthofStayinIC"])
df["LengthofStayinIC"] = encoder_LengthofStayinIC.transform(df["LengthofStayinIC"])

x = df[variable_cols]
y = df["time"]
actime = df['Operatingtime']
x_train, x_test, y_train, y_test  = train_test_split(x, y, test_size=0.2)

# rfc_100 = RandomForestClassifier(n_estimators=100,random_state=0)
# rfc_100.fit(x_train,y_train)
# feature_scores = pd.Series(rfc_100.feature_importances_, index=x_train.columns).sort_values(ascending=False)
# print(feature_scores)

select_variable_cols = ['SurgeryType','Age','Euroscore1','LengthofStayinHospital','Surgeon','LengthofStayinIC','DayTipe','Anesthesiologist','Hypercholesterolemia','Hypertension']
x_train = x_train[select_variable_cols]
x_test = x_test[select_variable_cols]
z_test = x[select_variable_cols]

# Machine learining classification models
# Random Forest
rf_model = RandomForestClassifier(n_estimators=100,random_state=0)
rf_model.fit(x_train, y_train)
rf_y_pred = rf_model.predict(x_test)
rf_z_pred = rf_model.predict(z_test)
print('RF Accuracy:', accuracy_score(y_test, rf_y_pred))
print('RF Accuracy whole dataset', accuracy_score(y, rf_z_pred))
print('RF Precision:',precision_score(y_test,rf_y_pred,average='micro'))
print('RE Recall: ',recall_score(y_test,rf_y_pred,average='micro'))
print('RE Confusion Matrix: ' )
print(confusion_matrix(y_test,rf_y_pred))
print('')
print('RE Classification report:')
print(classification_report(y_test,rf_y_pred))

#Decision Tree
dt_model = DecisionTreeClassifier(random_state=0)
dt_model.fit(x_train, y_train)
dt_y_pred = dt_model.predict(x_test)
dt_z_pred = dt_model.predict(z_test)

print('DT Accuracy:', accuracy_score(y_test, dt_y_pred))
print('DT Accuracy whole dataset', accuracy_score(y, dt_z_pred))
print('DT Precision:',precision_score(y_test,dt_y_pred,average='micro'))
print('DT Recall: ',recall_score(y_test,dt_y_pred,average='micro'))
print('DT Confusion Matrix: ' )
print(confusion_matrix(y_test,dt_y_pred))
print('')
print('DT Classification report:')
print(classification_report(y_test,dt_y_pred))

#support vector machine
svc_model = SVC(C=100.0)
svc_model.fit(x_train, y_train)
svc_y_pred = svc_model.predict(x_test)
svc_z_pred = svc_model.predict(z_test)
print('DT Accuracy:', accuracy_score(y_test, svc_y_pred ))
print('DT Accuracy whole dataset', accuracy_score(y, svc_z_pred))
print('DT Precision:',precision_score(y_test,svc_y_pred ,average='micro'))
print('DT Recall: ',recall_score(y_test,svc_y_pred ,average='micro'))
print('DTConfusion Matrix: ' )
print(confusion_matrix(y_test,svc_y_pred ))
print('')
print('DTClassification report:')
print(classification_report(y_test,svc_y_pred))

#logistic regression
lr_model = LogisticRegression()
lr_model.fit(x_train, y_train)
lr_y_pred = lr_model.predict(x_test)
lr_z_pred = lr_model.predict(z_test)
print('LR Accuracy:', accuracy_score(y_test, lr_y_pred ))
print('LR Accuracy whole dataset', accuracy_score(y, lr_z_pred))
print('LR Precision:',precision_score(y_test,lr_y_pred ,average='micro'))
print('LR Recall: ',recall_score(y_test,lr_y_pred,average='micro'))
print('LR Confusion Matrix: ' )
print(confusion_matrix(y_test,lr_y_pred ))
print('')
print('LR Classification report:')
print(classification_report(y_test,lr_y_pred))



# In[32]:


df['dt_z_pred'] = dt_z_pred
df.to_csv('data1.csv')


# In[34]:


df['rf_z_pred'] = rf_z_pred
df.to_csv('data2.csv')


# In[35]:


df['svc_z_pred'] = svc_z_pred
df.to_csv('data3.csv')


# In[33]:


df['lr_z_pred'] = lr_z_pred
df.to_csv('data4.csv')


# In[ ]:




