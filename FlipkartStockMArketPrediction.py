#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# part 0 q1
import pandas as pd
#making data frame
data = pd.read_csv("test_DataScience.csv")
# printing the column name and their datatype
DataTypeSeries= data.dtypes
print(DataTypeSeries)


# In[ ]:


#Part 0 q2
import pandas as pd
dt=pd.read_csv("test_DataScience.csv")
#fetching the places in india from which the page was accessed.
print(dt[["Place_in_India"]])


# In[ ]:


data['Place_in_India'].value_counts()


# In[ ]:


import pandas as pd
from datetime import timedelta
df=pd.read_csv("test_DataScience.csv")
cols=["Month","Year"]
df['Date'] = df[cols].apply(lambda x: '-'.join(x.values.astype(str)), axis="columns")
df['Date']=pd.to_datetime(df['Date'])
df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%d/%m/%Y')
df = df[['Year', 'Month', 'Date', 'Laptop/Desktop', 'Type_of_Customers?','Coming from','Place_in_India','Level 1','Level 2','Level 3','Level 4']]
df.head()


# In[ ]:


import pandas as pd
df=pd.read_csv("test_DataScience.csv")
print(df.isnull().sum())    #no of missing values each column
print(df.isnull().sum().sum())  #no of missing values each column
df["Level 1"]=df["Level 1"].fillna(df["Level 1"].mean())
df["Level 2"]=df["Level 2"].fillna(df["Level 2"].mean())
print(df)


# In[ ]:


import pandas as pd
def GetMonthInInt(month):
    MonthInInts = pd.Series([1,2,3,4,5,6,7,8,9,10,11,12],index=['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'])
    return MonthInInts[month.lower()]
df=pd.read_csv("test_DataScience.csv")
df['B']= df['Month'].apply(GetMonthInInt)
print(df)
#df.head()


# In[ ]:


import pandas as pd
df=pd.read_csv("test_DataScience.csv")
df['Coming from'] = df['Coming from'].replace(['Came_From_LinkedIn', 'Landed_Directly'], ['LinkedIn', 'Direct_Traffic'])
df['E']=df['Coming from']
df


# In[37]:


import pandas as pd
#data=pd.read_csv("test_DataScience.csv")
def descriptive_stats(Year, Month, Laptop, Type_of_Customer,Coming_from, Place_in_India):
    return data={"Year":['2020','2014','2016','2021','2022','2018'],
         "Month":['June','July','Jan','March','Oct','Dec'],
          "Laptop":['Laptop','Desktop','Desktop','Laptop','Desktop','PC'],
          "Type_of_Customer":['Existing','Existing','Existing','Existing','New','New'],
           "Coming_from":['LinkedIN','Sources','LinkedIN','LinkedIN','LinkedIN','LinkedIN'],
           "Place_in_India":['Pune','Gujarat','Delhi','Mumbai','Solapur','Kolkata']}
    
df= pd.DataFrame(data)
df


# In[34]:


import pandas as pd
#data=pd.read_csv("test_DataScience.csv")
def descriptive_stats(Year, Month, Laptop, Type_of_Customer,Coming_from, Place_in_India):
    return data=dict{"Year":['2020','2014','2016','2021','2022','2018'],
         "Month":['June','July','Jan','March','Oct','Dec'],
          "Laptop":['Laptop','Desktop','Desktop','Laptop','Desktop','PC'],
          "Type_of_Customer":['Existing','Existing','Existing','Existing','New','New'],
           "Coming_from":['LinkedIN','Sources','LinkedIN','LinkedIN','LinkedIN','LinkedIN'],
           "Place_in_India":['Pune','Gujarat','Delhi','Mumbai','Solapur','Kolkata']}
    
df= pd.DataFrame()
df.append(data, ignore_index=True)
print(df.head())


# In[38]:


import pandas as pd
data={"Year":['2020','2014','2016','2021','2022','2018'],
         "Month":['June','July','Jan','March','Oct','Dec'],
          "Laptop":['Laptop','Desktop','Desktop','Laptop','Desktop','PC'],
          "Type_of_Customer":['Existing','Existing','Existing','Existing','New','New'],
           "Coming_from":['LinkedIN','Sources','LinkedIN','LinkedIN','LinkedIN','LinkedIN'],
           "Place_in_India":['Pune','Gujarat','Delhi','Mumbai','Solapur','Kolkata']}
df= pd.DataFrame(data)
df


# In[18]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
data={"Year":['2020','2014','2016','2021','2022','2018'],
         "Month":['June','July','Jan','March','Oct','Dec'],
          "Laptop":['Laptop','Desktop','Desktop','Laptop','Desktop','PC'],
          "Type_of_Customer":['Existing','Existing','Existing','Existing','New','New'],
           "Coming_from":['LinkedIN','Sources','LinkedIN','LinkedIN','LinkedIN','LinkedIN'],
           "Place_in_India":['Pune','Gujarat','Delhi','Mumbai','Solapur','Kolkata']}
index_labels=['r1','r2','r3','r4','r5','r6']
print("The shape of the dataframe is: ", df.shape)
#df.describe()
df = pd.DataFrame(data,index=index_labels)
df_mean = df["Year"].mean()
print(df_mean) #calculating mean
print(df.median()) #calculating median
df.std(axis = 1, skipna = True)

# Removing the outliers
def removeOutliers(df, Year):
    Q3 = np.quantile(df[Year], 0.75)
    Q1 = np.quantile(df[Year], 0.25)
    IQR = Q3 - Q1

print("IQR value for column %s is: %s" % (Year, IQR))
global outlier_free_list
global filtered_data

lower_range = Q1 - 1.5 * IQR
upper_range = Q3 + 1.5 * IQR
outlier_free_list = [x for x in data[Year] if (
(x > lower_range) & (x < upper_range))]
filtered_data = df.loc[data[Year].isin(outlier_free_list)]


for i in df.columns:
    if i == df.columns[0]:
        removeOutliers(df, i)
    else:
        removeOutliers(filtered_data, i)


# Assigning filtered data back to our original variable
df = filtered_data
print("Shape of data after outlier removal is: ", df.shape)





# In[23]:


df.applymap(np.isreal).all(1) #if all values are false then they are non-numeric.
df[~df.applymap(np.isreal).all(1)]
print(df.Place_in_India.unique())
print(df.Laptop.unique())
print(df.Coming_from.unique())
print(df.Type_of_Customer.unique())
print(pd.unique(df['Year']))


# In[33]:


import pandas as pd
import numpy as np
df=pd.read_csv("test_DataScience.csv")
df.sort_values(['Level 1','Place_in_India'],ascending = False).groupby('Level 1').head(5) 


# In[36]:


import pandas as pd
import numpy as np
df=pd.read_csv("test_DataScience.csv")
df.sort_values(['Level 4','Place_in_India'],ascending = True).groupby('Level 4').head(5) 


# In[57]:



df2 = df.groupby('Place_in_India').sum()
df2
df2['Sum of level 2/Sum of level 1'] = df2['Level 2']/df2['Level 1']
df2['Sum of level 3/Sum of level 1'] = df2['Level 3']/df2['Level 1']
df2['Sum of level 4/Sum of level 1'] = df2['Level 4']/df2['Level 1']
df2


# In[59]:


import pandas as pd
df=pd.read_csv('test_DataScience.csv')
df.plot( 'Month' , 'Level 2' )


# In[78]:


import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv('test_DataScience.csv')
plt.plot(df["Month"], df["Place_in_India"], df["Level 2"])
plt.show()


# In[94]:


import pandas as pd
import matplotlib.pyplot as plt 
df=pd.read_csv('test_DataScience.csv')
df = df.head()
data = pd.DataFrame(df, columns=["Month", "Level 2", "Place_in_India"])
# plot the dataframe
data.plot(x="Month", y=["Level 2", "Place_in_India"], kind="bar", figsize=(10, 10))
# print bar graph
plt.show()


# In[108]:


import pandas as pd
import matplotlib.pyplot as plt 
df=pd.read_csv('test_DataScience.csv')
#df = df.head()
data = pd.DataFrame(df, columns=["Month", "Level 1", "Laptop/Desktop"])
# plot the dataframe
data.plot(x="Month", y=["Level 1", "Laptop/Desktop"], kind="line", figsize=(10, 7))
# print bar graph
plt.show()


# In[9]:


import pandas as pd
import matplotlib.pyplot as plt 
df=pd.read_csv('test_DataScience.csv')
#df = df.head()
data = pd.DataFrame(df, columns=["Year", "Level 2", "Coming_from"])
# plot the dataframe
data.plot(x="Year", y=["Level 2", "Coming_from"], kind="line", figsize=(10, 7))
# print bar graph
plt.show()


# In[10]:


import pandas as pd
import matplotlib.pyplot as plt 
df=pd.read_csv('test_DataScience.csv')
#df = df.head()
data = pd.DataFrame(df, columns=["Year", "Level 1", "Level 2"])
# plot the dataframe
data.plot(x="Year", y=["Level 1", "Level 2"], kind="line", figsize=(10, 7))
# print bar graph
plt.show()


# In[11]:


import pandas as pd
import matplotlib.pyplot as plt 
df=pd.read_csv('test_DataScience.csv')
#df = df.head()
data = pd.DataFrame(df, columns=["Place-in_India", "Level 3"])
# plot the dataframe
data.plot(x="Place-in_India", y=["Level 3"], kind="line", figsize=(10, 7))
# print bar graph
plt.show()


# In[14]:


import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv('test_DataScience.csv')
plt.plot( df["Place_in_India"],df["Year"], df["Level 3"])
plt.show()


# In[27]:


import pandas as pd
def predict_future(Year='2020',Month='Jan',LaptopDesktop='Laptop',Type_of_Customers='New',Coming_from='Socialmedia',Place_in_India='Pune'):
    print(Year,Month,LaptopDesktop, Type_of_Customers, Coming_from, Place_in_India)
df=predict_future(2020,'Jan','Lap','New','Media','Hyderabad') 
df


# In[24]:


import numpy as np
import pandas as pd
df = pd.DataFrame(
    np.random.randint(10, size=(6,79)), 
    columns=list("Year, Month , Laptop/Desktop , Type_of_Customers? ,Coming from , Place_in_India")
)
df


# In[ ]:


import pandas as pd
def predict_future(Year='2020',Month='Jan',LaptopDesktop='Laptop',Type_of_Customers='New',Coming_from='Socialmedia',Place_in_India='Pune'):
    #print('Year','Month','LaptopDesktop', 'Type_of_Customers', 'Coming_from', 'Place_in_India')
    return predict_future()
df=predict_future(2020,Jan,Lap,New,Media,Hyderabad) 
df


# In[ ]:


import pandas as pd
def predict_future(Year='2020',Month='Jan',LaptopDesktop='Laptop',Type_of_Customers='New',Coming_from='Socialmedia',Place_in_India='Pune'):
    print('Year','Month','LaptopDesktop', 'Type_of_Customers', 'Coming_from', 'Place_in_India')
    return predict_future(2020,'Jan','Lap','New','Media','Hyderabad')
#df=predict_future(2020,Jan,Lap,New,Media,Hyderabad) 
#df
predict_future()


# In[ ]:


import pandas as pd
def predict_future(Year='2020',Month='Jan',LaptopDesktop='Laptop',Type_of_Customers='New',Coming_from='Socialmedia',Place_in_India='Pune'):
    #print('Year','Month','LaptopDesktop', 'Type_of_Customers', 'Coming_from', 'Place_in_India')
    return predict_future()
df=predict_future(2020,'Jan','Lap','New','Media','Hyderabad') 


# In[9]:


import pandas as pd
df=pd.read_csv('test_DataScience.csv')
def predict_future(Year='2020',Month='Jan',LaptopDesktop='Laptop',Type_of_Customers='New',Coming_from='Socialmedia',Place_in_India='Pune'):
    print(Year,Month,LaptopDesktop, Type_of_Customers, Coming_from, Place_in_India)
    #return predict_future()
predict_future(2020,'Jan','Lap','New','Media','Hyderabad')
predict_future(2020,'Jan','Lap','New','Media','Hyderabad')
predict_future(2020,'Jan','Lap','New','Media','Pune')
predict_future(2020,'Jan','PC','Existing','Media','Solapur')
predict_future(2020,'Jan','Lap','New','LinkedIn','Hyderabad')
predict_future(2020,'Jan','Lap','New','Media','Hyderabad')
df=predict_future()
df


# In[20]:


import pandas as pd
df=pd.read_csv('test_DataScience.csv')
df['LaptopDesktop']=df['Laptop/Desktop']
df.rename(columns = {'Laptop/Desktop':'LaptopDesktop'}, inplace = True)
df.rename(columns = {'Type_of_Customers?':'Type_of_Customers'}, inplace = True)
df.rename(columns = {'Coming from':'Coming_from'}, inplace = True)
  
def predict_future(Year='2020',Month='Jan',LaptopDesktop='Laptop',Type_of_Customers='New',Coming_from='Socialmedia',Place_in_India='Pune'):
        return df.predict_future
df.Year='2020'
df.Month='jan'
df.LaptopDesktop='Laptop'
df.Type_of_Customers='New'
df.Coming_from='Socialmedia'
df.Place_in_India='Pune'
df


# In[24]:


import pandas as pd
df=pd.read_csv('test_DataScience.csv')
df['LaptopDesktop']=df['Laptop/Desktop']
df.rename(columns = {'Laptop/Desktop':'LaptopDesktop'}, inplace = True)
df.rename(columns = {'Type_of_Customers?':'Type_of_Customers'}, inplace = True)
df.rename(columns = {'Coming from':'Coming_from'}, inplace = True)
  
def descriptive_stats(Year='2020',Month='Jan',LaptopDesktop='Laptop',Type_of_Customers='New',Coming_from='Socialmedia',Place_in_India='Pune'):
        return df.predict_future
df.Year='2020'
df.Month='jan'
df.LaptopDesktop='Laptop'
df.Type_of_Customers='New'
df.Coming_from='Socialmedia'
df.Place_in_India='Pune'
df
#index_labels=['r1','r2','r3','r4','r5','r6']
print("The shape of the dataframe is: ", df.shape)
#df.describe()
dfnew = pd.DataFrame(df,index=index_labels)
df_mean = dfnew["Year"].mean()
print(df_mean) #calculating mean
print(dfnew.median()) #calculating median
dfnew.std(axis = 1, skipna = True)

# Removing the outliers
def removeOutliers(dfnew, Year):
    Q3 = np.quantile(dfnew[Year], 0.75)
    Q1 = np.quantile(dfnew[Year], 0.25)
    IQR = Q3 - Q1

print("IQR value for column %s is: %s" % (Year, IQR))
global outlier_free_list
global filtered_data

lower_range = Q1 - 1.5 * IQR
upper_range = Q3 + 1.5 * IQR
outlier_free_list = [x for x in data[Year] if (
(x > lower_range) & (x < upper_range))]
filtered_data = df.loc[data[Year].isin(outlier_free_list)]


for i in dfnew.columns:
    if i == dfnew.columns[0]:
        removeOutliers(df, i)
    else:
        removeOutliers(filtered_data, i)


# Assigning filtered data back to our original variable
dfnew = filtered_data
print("Shape of data after outlier removal is: ", dfnew.shape)



# In[28]:


import pandas as pd
import numpy as np
df=pd.read_csv('test_DataScience.csv')
df['LaptopDesktop']=df['Laptop/Desktop']
df.rename(columns = {'Laptop/Desktop':'LaptopDesktop'}, inplace = True)
df.rename(columns = {'Type_of_Customers?':'Type_of_Customers'}, inplace = True)
df.rename(columns = {'Coming from':'Coming_from'}, inplace = True)
  
def predict_future(Year='2020',Month='Jan',LaptopDesktop='Laptop',Type_of_Customers='New',Coming_from='Socialmedia',Place_in_India='Pune'):
        return df.predict_future
df.Year='2020'
df.Month='jan'
df.LaptopDesktop='Laptop'
df.Type_of_Customers='New'
df.Coming_from='Socialmedia'
df.Place_in_India='Pune'
df
df.applymap(np.isreal).all(1)  #if all values are false then they are non-numeric.
df[~df.applymap(np.isreal).all(1)]
print(df.Place_in_India.unique())
print(df.LaptopDesktop.unique())
print(df.Coming_from.unique())
print(df.Type_of_Customers.unique())
print(pd.unique(df['Year']))


# In[30]:


import pandas as pd
import numpy
df=pd.read_csv('test_DataScience.csv')
df['LaptopDesktop']=df['Laptop/Desktop']
df.rename(columns = {'Laptop/Desktop':'LaptopDesktop'}, inplace = True)
df.rename(columns = {'Type_of_Customers?':'Type_of_Customers'}, inplace = True)
df.rename(columns = {'Coming from':'Coming_from'}, inplace = True)
  
def predict_future(Year='2020',Month='Jan',LaptopDesktop='Laptop',Type_of_Customers='New',Coming_from='Socialmedia',Place_in_India='Pune'):
        return df.predict_future
df.Year='2020'
df.Month='jan'
df.LaptopDesktop='Laptop'
df.Type_of_Customers='New'
df.Coming_from='Socialmedia'
df.Place_in_India='Pune'
df

#from statsmodels.tsa.ar_model import AutoRegResults
#import numpy
# load model
model = df.load('test_DataScience.csv')
df = numpy.load('df.npy')
last_ob = numpy.load('df.npy')
# make prediction
predictions = model.predict(start=len(df), end=len(df))
# transform prediction
yhat = predictions[0] + last_ob[0]
print('Prediction: %f' % yhat)


# In[32]:


import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from StringIO import StringIO

data = StringIO("""
Year Carbon June
2000 6727 20.386
2001 6886 20.445
2002 6946 20.662
2003 7367 20.343
2004 7735 20.242
2005 8025 20.720
2006 8307 20.994
2007 8488 20.661
2008 8738 20.657
2009 8641 20.548
2010 9137 21.027
2011 9508 20.915
2012 9671 21.172
""")

# Model training
df = pd.read_table(data, index_col=0, sep='\s+')
Y_train = df['June']
X_train = df['Carbon']
X_train = sm.add_constant(X_train) # add this to your code
model = sm.OLS(Y_train, X_train)
results = model.fit()

# Prediction of future values
future_carbon = range(9700, 10000, 50)
X_pred = pd.DataFrame(data=future_carbon, columns=['Carbon'])
X_pred = sm.add_constant(X_pred)
prediction = model.predict(results.params, X_pred)

# Plot
plt.figure()
plt.plot(X_train['Carbon'], model.predict(results.params), '-r', label='Linear model')
plt.plot(X_pred['Carbon'], prediction, '--r', label='Linear prediction')
plt.scatter(df['Carbon'], df['June'], label='data')
plt.xlabel('Carbon')
plt.ylabel('June temperature')
plt.legend()
plt.show()


# In[6]:


import pandas as pd
import numpy
#import StringIO
import matplotlib
df=pd.read_csv('test_DataScience.csv')
df['LaptopDesktop']=df['Laptop/Desktop']
df.rename(columns = {'Laptop/Desktop':'LaptopDesktop'}, inplace = True)
df.rename(columns = {'Type_of_Customers?':'Type_of_Customers'}, inplace = True)
df.rename(columns = {'Coming from':'Coming_from'}, inplace = True)
  
def predict_future(Year='2020',Month='Jan',LaptopDesktop='Laptop',Type_of_Customers='New',Coming_from='Socialmedia',Place_in_India='Pune'):
        return df.predict_future
df.Year='2020'
df.Month='jan'
df.LaptopDesktop='Laptop'
df.Type_of_Customers='New'
df.Coming_from='Socialmedia'
df.Place_in_India='Pune'
df

#read data from extracted csv
steps=pd.read_csv('test_DataScience.csv')
#convert start date into time format
steps['Level_4']=pd.to_datetime(steps['Level 4'].datetime
#Aggregate data into weekly sum
sample=steps[['Level_4','value']]
weekly=sample.resample('W', on='date').sum()
#visualize weekly data
weekly.plot(figsize=(15, 6))
plt.show()


# In[12]:


import pandas as pd
import numpy
df=pd.read_csv('test_DataScience.csv')
df['LaptopDesktop']=df['Laptop/Desktop']
df.rename(columns = {'Laptop/Desktop':'LaptopDesktop'}, inplace = True)
df.rename(columns = {'Type_of_Customers?':'Type_of_Customers'}, inplace = True)
df.rename(columns = {'Coming from':'Coming_from'}, inplace = True)
  
def predict_future(Year='2020',Month='Jan',LaptopDesktop='Laptop',Type_of_Customers='New',Coming_from='Socialmedia',Place_in_India='Pune'):
        return df.predict_future
df.Year='2020'
df.Month='jan'
df.LaptopDesktop='Laptop'
df.Type_of_Customers='New'
df.Coming_from='Socialmedia'
df.Place_in_India='Pune'
df

#df[(df['Level 4'].dt.month == 1) & (df['Level 4'].dt.day == 1)].mean()
df = df.groupby(by=[df.index.Year, df.index.Level4]).mean() 


# In[15]:


import numpy as np
from sklearn.model_selection import train_test_split 
import pandas as pd
df=pd.read_csv('test_DataScience.csv')
df['LaptopDesktop']=df['Laptop/Desktop']
df.rename(columns = {'Laptop/Desktop':'LaptopDesktop'}, inplace = True)
df.rename(columns = {'Type_of_Customers?':'Type_of_Customers'}, inplace = True)
df.rename(columns = {'Coming from':'Coming_from'}, inplace = True)
  
def predict_future(Year='2020',Month='Jan',LaptopDesktop='Laptop',Type_of_Customers='New',Coming_from='Socialmedia',Place_in_India='Pune'):
        return df.predict_future
df.Year='2020'
df.Month='jan'
df.LaptopDesktop='Laptop'
df.Type_of_Customers='New'
df.Coming_from='Socialmedia'
df.Place_in_India='Pune'
df
 
#Separating the dependent and independent data variables into two data frames.
X = df.drop(['Level 1'],axis=1) 
Y = df['Level 1']
 
# Splitting the dataset into 80% training data and 20% testing data.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.20, random_state=0)
 
#Defining MAPE function
def MAPE(Y_actual,Y_Predicted):
    mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
    return mape


#Building the Linear Regression Model
from sklearn.linear_model import LinearRegression
linear_model = LinearRegression().fit(X_train , Y_train)
 
#Predictions on Testing data
LR_Test_predict = linear_model.predict(X_test) 
 
# Using MAPE error metrics to check for the error rate and accuracy level
LR_MAPE= MAPE(Y_test,LR_Test_predict)
print("MAPE: ",LR_MAPE)


# In[ ]:




