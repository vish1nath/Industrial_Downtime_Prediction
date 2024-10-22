
# coding: utf-8

# In[425]:

#get_ipython().magic(u'matplotlib inline')
import pandas as pd
import numpy as np
#pd.options.display.max_columns=100


# # Loading the Indicator Reading Dataset 

# In[426]:

Indicator= pd.read_excel('Indicator.xlsx', index_col=None,parse_dates=
                           ['Date and Time Collected','Date And Time Uploaded'])


# Filtering the Dataset to fetch only the HAGC Assets reading

# In[427]:

HAGInd=Indicator[Indicator['Asset Number']=='544-630']
I=HAGInd[['Indicator Name','Indicator Reading','Indicator State Name','Alarm Type Name','Date and Time Collected']]


# Formating the Date and slicing the data to fetch only Indicator Name and its Readings.On some dates the Indicator readings are taken more then once,so I removed the duplicates and took the latest reading of that date

# In[428]:

I['Date']=I['Date and Time Collected'].dt.date
I=I[['Date','Indicator Name','Indicator Reading']]
I=I.drop_duplicates(['Date','Indicator Name'], take_last=True)


# # Pivoting the Data 
# Converting the Indicator Names into columns and its readings as rows

# In[429]:

Reading1=I.pivot(index='Date',columns='Indicator Name',values='Indicator Reading')


# Formating the reading values so that it can be converted into Integer

# In[430]:


Reading1['HAGC Fluid LOSS Calculation (E Bin, W Bin, Refill Bins)']=Reading1['HAGC Fluid LOSS Calculation (E Bin, W Bin, Refill Bins)'].str.split().str[0].str.replace(',','')
Reading1['HAGC 3 Day Average Fluid Loss']=Reading1['HAGC 3 Day Average Fluid Loss'].str.split().str[0].str.replace(',','')
Reading1['HAGC Hydraulic Fluid Temperature']=Reading1['HAGC Hydraulic Fluid Temperature'].str.split().str[0].str.replace(',','')
Reading1['HAGC Hydraulic Tank Level']=Reading1['HAGC Hydraulic Tank Level'].str.split().str[0].str.replace(',','')
Reading1['HAGC Total Monthly Fluid Loss']=Reading1['HAGC Total Monthly Fluid Loss'].str.split().str[0].str.replace(',','')
Reading1['HAGC Total Weekly Fluid Loss']=Reading1['HAGC Total Weekly Fluid Loss'].str.split().str[0].str.replace(',','')


# In[431]:

Reading1[['HAGC Fluid LOSS Calculation (E Bin, W Bin, Refill Bins)','HAGC Total Weekly Fluid Loss',
   'HAGC Total Monthly Fluid Loss','HAGC Hydraulic Tank Level','HAGC Hydraulic Fluid Temperature'
   ,'HAGC 3 Day Average Fluid Loss']]=Reading1[['HAGC Fluid LOSS Calculation (E Bin, W Bin, Refill Bins)','HAGC Total Weekly Fluid Loss',
   'HAGC Total Monthly Fluid Loss','HAGC Hydraulic Tank Level','HAGC Hydraulic Fluid Temperature'
   ,'HAGC 3 Day Average Fluid Loss']].convert_objects(convert_numeric=True)


# # Filling the NULL values
# The Tank Level and Fluid Temperature are having null values,After seeing the timeseries plots for these two variables,I have decided to use the interpolate these values

# In[432]:

Reading1['HAGC Hydraulic Tank Level']=Reading1['HAGC Hydraulic Tank Level'].interpolate(method='time')
Reading1['HAGC Hydraulic Fluid Temperature']=Reading1['HAGC Hydraulic Fluid Temperature'].interpolate(method='time')


# Filling the null values for Daily Fluid loss.After checking the time series graphs for daily loss,I do not see any sign of auto-correlation between the data and the value is reverting back to mean.hence I used the mean to fill the nulls

# In[433]:

Reading1['HAGC Fluid LOSS Calculation (E Bin, W Bin, Refill Bins)']=Reading1['HAGC Fluid LOSS Calculation (E Bin, W Bin, Refill Bins)'].fillna(value=Reading1['HAGC Fluid LOSS Calculation (E Bin, W Bin, Refill Bins)'].mean())


# To find the Weekly ,Monthly and 3 Day average I have used the moving averages and moving sum function.

# In[434]:

Reading1['HAGC Total Weekly Fluid Loss']=pd.rolling_sum(Reading1['HAGC Fluid LOSS Calculation (E Bin, W Bin, Refill Bins)'],7)
Reading1['HAGC Total Weekly Fluid Loss']=Reading1['HAGC Total Weekly Fluid Loss'].fillna(value=Reading1['HAGC Total Weekly Fluid Loss'].mean())
Reading1['HAGC Total Monthly Fluid Loss']=pd.rolling_sum(Reading1['HAGC Fluid LOSS Calculation (E Bin, W Bin, Refill Bins)'],30)
Reading1['HAGC Total Monthly Fluid Loss']=Reading1['HAGC Total Monthly Fluid Loss'].fillna(value=Reading1['HAGC Total Monthly Fluid Loss'].mean())
Reading1['HAGC 3 Day Average Fluid Loss']=pd.rolling_mean(Reading1['HAGC Fluid LOSS Calculation (E Bin, W Bin, Refill Bins)'],3)
Reading1['HAGC 3 Day Average Fluid Loss']=Reading1['HAGC 3 Day Average Fluid Loss'].fillna(value=Reading1['HAGC 3 Day Average Fluid Loss'].mean())


# I used back fill method to fill the nulls for other filter readings.I assume the next immediate reading availble is the closest to it.Since HAGC Hydraulic fluid has only 5 non null values,I am deleting this field.

# In[435]:

Reading1=Reading1.drop(['HAGC Hydraulic Fluid'],axis=1)


# In[436]:

Reading1=Reading1.fillna(method='bfill')


# In[437]:

Reading1=Reading1.loc['2012']


# # Loading the Downtime dataset

# In[438]:

Down=pd.read_excel('Downtime.xlsx', index_col=None,parse_dates=['Downtime Started','Downtime Ended'])
HAGCDown=Down[Down['ACD Asset Number']=='544-630']


# Filtering the year

# In[439]:

D=HAGCDown[HAGCDown['Downtime Started'].dt.year==2012]


# # Merging the Datasets
# Fetching the Downtime started dates to merge with indicator data and marking the two days before the failure to predict the failure in following 48hrs period

# In[440]:

z=Reading1.index.isin(D['Downtime Started'].dt.date)
for i in range(len(z)):
    if(z[i]==True):
        #z[i-6]=True
        #z[i-5]=True
        #z[i-3]=True
        z[i-1]=True
        z[i-2]=True


# Adding the dates column as boolean(target label) to identify the failures

# In[441]:

Reading1['Failure']=z.astype(int)


# # Creating Dummies to filter variables
# 
# I am converting all the categorical variables to binary using dummies method.This is an essential step to run the machine learning algorithms in Scikit learn

# In[442]:

hagc12=pd.get_dummies(Reading1['HAGC 1 & 2 Return Filter Condition'],prefix='HAGC12')
hagc34=pd.get_dummies(Reading1['HAGC 3 &4 Return Filter Condition'],prefix='HAGC34')
hagc78=pd.get_dummies(Reading1['HAGC 7 & 8 Return Filter Condition'],prefix='HAGC78')
URCC=pd.get_dummies(Reading1['HAGC Hydraulic UR Bin CCJensen'],prefix='URCC')
tankCC=pd.get_dummies(Reading1['HAGC Hydraulic tank CCJensen'],prefix='tankCC')
hagc1=pd.get_dummies(Reading1['HAGC Pump #1 Filter Condition'],prefix='HAGC1')
hagc2=pd.get_dummies(Reading1['HAGC Pump #2 Filter Condition'],prefix='HAGC2')
hagc3=pd.get_dummies(Reading1['HAGC Pump #3 Filter Condition'],prefix='HAGC3')
hagc4=pd.get_dummies(Reading1['HAGC Pump #4 Filter Condition'],prefix='HAGC4')
hagc5=pd.get_dummies(Reading1['HAGC Pump #5 Filter Condition'],prefix='HAGC5')
hagc6=pd.get_dummies(Reading1['HAGC Pump #6 Filter Condition'],prefix='HAGC6')
hagc7=pd.get_dummies(Reading1['HAGC Pump #7 Filter Condition'],prefix='HAGC7')
hagc8=pd.get_dummies(Reading1['HAGC Pump #8 Filter Condition'],prefix='HAGC8')
ss=pd.get_dummies(Reading1['HAGC Side Stream Filter Condition'],prefix='SS')


# Concatenating the dummy fields and dropping the original filter condition columns 

# In[443]:

Reading2=pd.concat([Reading1,hagc12,hagc34,hagc78,URCC,tankCC,hagc1,hagc2,hagc3,hagc4,hagc5,hagc6,hagc7,hagc8,ss],axis=1)


# In[444]:

Reading2=Reading2.drop(['HAGC 1 & 2 Return Filter Condition','HAGC 3 &4 Return Filter Condition',
           'HAGC 7 & 8 Return Filter Condition','HAGC Hydraulic UR Bin CCJensen','HAGC Hydraulic tank CCJensen',
           'HAGC Pump #1 Filter Condition','HAGC Pump #2 Filter Condition','HAGC Pump #3 Filter Condition',
           'HAGC Pump #4 Filter Condition','HAGC Pump #5 Filter Condition','HAGC Pump #6 Filter Condition',
           'HAGC Pump #7 Filter Condition','HAGC Pump #8 Filter Condition','HAGC Side Stream Filter Condition'],
          axis=1)





# Saving the transformed data set to run the models

# In[445]:

Reading2.to_csv('data1.csv')


# In[ ]:




# In[ ]:



