#!/usr/bin/env python
# coding: utf-8

# # Lending Club Case Study

# In[64]:


# import the libraries

import pandas as pd
import numpy as np

#pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
#pd.set_option('display.width', 1000)


# In[ ]:





# ### Loading the Dataframe

# In[65]:


#read the dataset and check the first five rows
#df_original = pd.read_excel("loan.xlsx", header=None, skiprows=7 ,sheet_name="C-08")
df_original = pd.read_csv("loan.csv")
df_original.head()


# ### Lets get some info 

# In[66]:


df_original.info()


# In[67]:


df_original.describe()


# In[115]:


df_original.isnull().sum()


# ## Lets Work on Getting required columns 

# #### By observation of data in given CSV , i found below columns which can be used to estimate the defaulters 

# In[68]:


col_needed  = ['id', 
               'loan_amnt', 
               'loan_status',  # study on Charged Off
               'funded_amnt', 
               'verification_status', # verified is better
               'application_type', #  no need
               'purpose', # 
               'chargeoff_within_12_mths', #  Filter only these to study
               'dti', # lower is better
               'delinq_2yrs', # lower is better
               'mort_acc', # lower is better
               'mths_since_last_delinq', # lower is better
               'mths_since_last_major_derog', # lower is better
               'bc_util', # lower is better
               'home_ownership', # own is least risk, Rent is medium, Mortgage is hig risk
               'max_bal_bc', # higher is better
               'emp_length', # 
               'delinq_amnt', # lower is better
               'pub_rec', # lower is better
               'revol_util', #
               'pub_rec_bankruptcies', # lower is better
               'acc_now_delinq', # lower is better
               'annual_inc', # higher is better
               'avg_cur_bal'] # higher is better

# get the required DF only
df_cols = df_original[col_needed]


# ### Lets one by one check each Column from the dataset

# In[69]:


#check unique value counts in column
df_cols['loan_status'].value_counts() # looks proper


# In[70]:


df_cols['verification_status'].value_counts() # good to have


# In[71]:


#df_cols['application_type'].value_counts() # only individual type is presen, Col no need
#df_cols['chargeoff_within_12_mths'].value_counts() # 0.0 value so no need col
df_cols['purpose'].value_counts() #  good to have


# In[72]:


df_cols['delinq_2yrs'].value_counts() ##  good to have


# In[73]:


df_cols['mort_acc'].value_counts() ##  no value , hence no need
df_cols['mths_since_last_delinq'].value_counts() ## Need this Col


# In[74]:


df_cols['mths_since_last_major_derog'].value_counts() ## no need 
df_cols['bc_util'].value_counts()## no need 
df_cols['home_ownership'].value_counts()## need this Col


# In[75]:


df_cols['max_bal_bc'].value_counts()#no need 
df_cols['emp_length'].value_counts()# need this and Cleaning needed 


# In[76]:


df_cols['dti'].value_counts() # good to have


# In[77]:


df_cols['pub_rec'].value_counts() ## good to have 


# In[78]:


df_cols['pub_rec_bankruptcies'].value_counts() ## Good to have 


# In[79]:


df_cols['acc_now_delinq'].value_counts() ## no use to keep col
df_cols['annual_inc'].value_counts() ##  good to keep


# In[80]:


df_cols['avg_cur_bal'].value_counts() ##  no use to keep


# In[81]:


df_cols['delinq_amnt'].value_counts()


# In[82]:


df_cols['revol_util'].value_counts()


# ### Lets Keep only required Cols

# In[83]:


col_needed2  = ['id', 
                'loan_amnt', 
                'loan_status',  # study on Charged Off
                'funded_amnt', 
                'verification_status', # verified is better
                #'application_type', #  no need
                'purpose', # 
                #'chargeoff_within_12_mths', #  Filter only these to study
                'dti', # lower is better
                'delinq_2yrs', # lower is bette
                #'mort_acc', # lower is better
                'mths_since_last_delinq', # lower is better
                #'mths_since_last_major_derog', # lower is better
                #'bc_util', # lower is better
                'home_ownership', # own is least risk, Rent is medium, Mortgage is hig risk
                #'max_bal_bc', # higher is better
                'emp_length', # 
                'pub_rec', # lower is better
                'revol_util', #
                'pub_rec_bankruptcies', # lower is better
                #'acc_now_delinq', # lower is better
                'annual_inc', # higher is better
                ##'avg_cur_bal'
               ] # higher is better

# get the required DF only
df_cols2 = df_original[col_needed2]


# In[84]:


## lets have a look at rows & cols once again
df_cols2.head()


# In[85]:


df_cols2.info()


# In[ ]:





# ### Data Cleaning, imputing and standardising values in Columns

# In[86]:


#df_cols2['loan_status'] = df_cols2['loan_status'].apply(lambda x: x.strip() )
#df_cols2['mths_since_last_delinq'].isna().sum()

## add 0 for filling up to the whole original DF  / imputing
df_original['mths_since_last_delinq'] = df_original['mths_since_last_delinq'].apply(lambda x: 0.0 if pd.isna(x) else x )

#df_cols2['mths_since_last_delinq'] = df_cols2['mths_since_last_delinq'].apply(lambda x: 0 if pd.isna(x) else x )


# In[87]:


# cleaning emp_length  col value and imputing 
#df_cols['emp_length'].value_counts() ###  get the  value counts 

# after unique value count, we have to remove strip "years" abd "+"
# set 0 for < 1 year or NAN
df_original['emp_length'] = df_original['emp_length'].apply(lambda x: '0' if ( (x == '< 1 year') or pd.isna(x) ) else x )

# Next filter all non-numeric value from all cells 
df_original['emp_length'] = df_original['emp_length'].apply(lambda x: int(str(x).strip('year').strip('years').strip().strip('+')))


# In[88]:


df_original['emp_length'].value_counts()


# In[89]:


# Clean and impute pub_rec_bankruptcies column
#df_original['pub_rec_bankruptcies'].isna().sum()

# set 0 for < 1 year or NAN
df_original['pub_rec_bankruptcies'] = df_original['pub_rec_bankruptcies'].apply(lambda x: 0 if pd.isna(x) else x )


# In[90]:



df_original['revol_util'] = df_original['revol_util'].apply(lambda x: float(str(x).strip().strip('%').strip() ) )
df_original['revol_util'] = df_original['revol_util'].apply(lambda x: 0.0 if pd.isna(x) else x )
df_original['revol_util'].value_counts()
#df_original['revol_util'].isna().sum()


# In[91]:


df_cols2 = df_original[col_needed2]
df_cols2.info()


# In[92]:


# we need to study pattern of Defaulters only as per the Problem Statement i.e "Charged Off" only
df_cols3 = df_cols2[df_cols2['loan_status'] !='Current']

df_cols3.info()


# In[ ]:





# ### We are all set for  Performing Analysis 

# In[93]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:





# In[94]:


# lets study each combination of column with. Filter out loan_status rows 
#df_with_loan_status  = df_cols[].groupby(by="")

##  loan_status VS loan_amnt
df_g1 = df_cols3[['loan_status','loan_amnt']].groupby(by="loan_status").median()

#df_g1 = df_cols3['loan_amnt'].value_counts()
plt.plot(df_g1)
plt.xlabel('loan_status')
plt.ylabel('loan_amnt')
plt.title('loan_status VS loan_amnt')
plt.show()

#plt.boxplot(df_g1)
#plt.show()


# In[95]:


##  loan_status VS funded_amnt

df_g2 = df_cols3[['loan_status','funded_amnt']].groupby(by='loan_status').median()

plt.plot(df_g2)
plt.xlabel('loan_status')
plt.ylabel('funded_amnt')
plt.title('loan_status VS funded_amnt')
plt.show()


# In[96]:


##  loan_status VS dti

df_g3 = df_cols3[['loan_status','dti']].groupby(by='loan_status').median()

plt.plot(df_g3)
plt.xlabel('loan_status')
plt.ylabel('DTI')
plt.title('loan_status VS DTI')
plt.show()


# In[97]:


##  loan_status VS delinq_2yrs

df_g4 = df_cols3[['loan_status','delinq_2yrs']].groupby(by='loan_status').median()

plt.plot(df_g4)
plt.xlabel('loan_status')
plt.ylabel('delinq_2yrs')
plt.title('loan_status VS delinq_2yrs')
plt.show()

######   revisit ######


# In[98]:


##  loan_status VS delinq_2yrs

##  lets plot grph first for Fully paid 
df_d4_tmp = df_cols3[df_cols3['loan_status'] =='Fully Paid']
df_g6_1 = df_d4_tmp[['delinq_2yrs']].value_counts()
ax_1 = df_g6_1.plot(kind='barh')
plt.ylabel("delinq_2yrs", labelpad=14)
plt.xlabel("Counts", labelpad=14)
plt.title("loan_status,Fully Paid VS delinq_2yrs", y=1.02)
ax_1.set_xscale('log')


# In[99]:


##  loan_status VS delinq_2yrs

##  lets plot grph now for charged off
df_d4_tmp = df_cols3[df_cols3['loan_status'] =='Charged Off']
df_g6_2 = df_d4_tmp[['delinq_2yrs']].value_counts()
ax_2 = df_g6_2.plot(kind='barh')
plt.ylabel("delinq_2yrs", labelpad=14)
plt.xlabel("Counts", labelpad=14)
plt.title("loan_status,Fully Paid VS delinq_2yrs", y=1.02)
ax_2.set_xscale('log')


# In[100]:


## remove the outlier 

plt.boxplot(df_d4_tmp[['delinq_2yrs']])

plt.show()

df_d4_tmp[['delinq_2yrs']].describe()


# In[ ]:





# In[101]:


##  loan_status VS mths_since_last_delinq

df_g5 = df_cols3[['loan_status','mths_since_last_delinq']].groupby(by='loan_status').median()

plt.plot(df_g5)
plt.xlabel('loan_status')
plt.ylabel('mths_since_last_delinq')
plt.title('loan_status VS mths_since_last_delinq')
plt.show()


# In[102]:


##  loan_status VS home_ownership

##  lets plot grph first for Fully paid 
df_d6_tmp = df_cols2[df_cols2['loan_status'] =='Fully Paid']
df_g6_1 = df_d6_tmp[['home_ownership']].value_counts()
ax_1 = df_g6_1.plot(kind='barh')
plt.ylabel("home_ownership", labelpad=14)
plt.xlabel("Counts of Fully Paid", labelpad=14)
plt.title("loan_status,Fully Paid VS home_ownership", y=1.02)
ax_1.set_xscale('log')



#plt.barplot(df_g6)
#sns.barplot(x = 'home_ownership', y = 'Freq', data = df_g6)
#plt.xlabel('loan_status')
#plt.ylabel('mths_since_last_delinq')
#plt.title('loan_status VS mths_since_last_delinq')
#plt.show()


# In[103]:


##  lets plot grph now for Charged Off
df_d6_tmp = df_cols2[df_cols2['loan_status'] =='Charged Off']
df_g6_2 = df_d6_tmp[['home_ownership']].value_counts()
ax_2 = df_g6_2.plot(kind='barh')
plt.ylabel("home_ownership", labelpad=14)
plt.xlabel("Counts of Charged Off", labelpad=14)
plt.title("loan_status,Charged Off VS home_ownership", y=1.02)
ax_2.set_xscale('log')


# In[ ]:





# In[104]:


##  loan_status VS emp_length

df_g7 = df_cols3[['loan_status','emp_length']].groupby(by='loan_status').median()

print(df_g7)

plt.plot(df_g7)
plt.xlabel('loan_status')
plt.ylabel('emp_length')
plt.title('loan_status VS emp_length')
plt.show()


# In[105]:


##  loan_status VS pub_rec

df_g8 = df_cols3[['loan_status','pub_rec']].groupby(by='loan_status').median()

print(df_g8)

plt.plot(df_g8)
plt.xlabel('loan_status')
plt.ylabel('pub_rec')
plt.title('loan_status VS pub_rec')
plt.show()


# In[106]:


##  loan_status VS pub_rec_bankruptcies

df_g9 = df_cols3[['loan_status','pub_rec_bankruptcies']].groupby(by='loan_status').mean()

print(df_g9)

plt.plot(df_g9)
plt.xlabel('loan_status')
plt.ylabel('pub_rec_bankruptcies')
plt.title('loan_status VS pub_rec_bankruptcies')
plt.show()


# In[107]:


##  loan_status VS annual_inc

df_g10 = df_cols3[['loan_status','annual_inc']].groupby(by='loan_status').mean()

print(df_g10)

plt.plot(df_g10)
plt.xlabel('loan_status')
plt.ylabel('annual_inc')
plt.title('loan_status VS annual_inc')
plt.show()


# In[108]:


##  loan_status VS purpose

##  lets plot grph first for Fully paid 
df_d11_tmp = df_cols2[df_cols2['loan_status'] =='Fully Paid']
df_g11_1 = df_d11_tmp[['purpose']].value_counts()
ax_11 = df_g11_1.plot(kind='barh')
plt.ylabel("purpose", labelpad=14)
plt.xlabel("Counts", labelpad=14)
plt.title("loan_status,Fully Paid VS purpose", y=1.02)
ax_11.set_xscale('log')


# In[109]:


##  loan_status VS purpose

##  lets plot grph first for Fully paid 
df_d11_tmp = df_cols2[df_cols2['loan_status'] =='Charged Off']
df_g11_2 = df_d11_tmp[['purpose']].value_counts()
ax_11 = df_g11_2.plot(kind='barh')
plt.ylabel("purpose", labelpad=14)
plt.xlabel("Counts", labelpad=14)
plt.title("loan_status,charged Off VS purpose", y=1.02)
ax_11.set_xscale('log')


# In[110]:


##  loan_status VS verification_status

##  lets plot grph first for Fully paid 
df_d12_tmp = df_cols2[df_cols2['loan_status'] =='Fully Paid']
df_g12_1 = df_d12_tmp[['verification_status']].value_counts()
ax_12 = df_g12_1.plot(kind='barh')
plt.ylabel("verification_status", labelpad=14)
plt.xlabel("Counts of Fully paid", labelpad=14)
plt.title("loan_status,Fully Paid VS verification_status", y=1.02)
ax_11.set_xscale('log')


# In[111]:


##  loan_status VS verification_status

##  lets plot grph for charged Off
df_d12_tmp = df_cols2[df_cols2['loan_status'] =='Charged Off']
df_g12_2 = df_d12_tmp[['verification_status']].value_counts()
ax_12 = df_g12_2.plot(kind='barh')
plt.ylabel("verification_status", labelpad=14)
plt.xlabel("Counts of Charged Off", labelpad=14)
plt.title("loan_status,charged Off VS verification_status", y=1.02)
ax_12.set_xscale('log')


# In[112]:


##  loan_status VS revol_util

df_g13 = df_cols3[['loan_status','revol_util']].groupby(by='loan_status').mean()

print(df_g13)

plt.plot(df_g13)
plt.xlabel('loan_status')
plt.ylabel('revol_util')
plt.title('loan_status VS revol_util')
plt.show()


# In[113]:


##  loan_status VS emp_length

##  lets plot grph first for Fully paid 
df_d14_tmp = df_cols2[df_cols2['loan_status'] =='Fully Paid']
df_g14_1 = df_d14_tmp[['emp_length']].value_counts()
ax_1 = df_g14_1.plot(kind='barh')
plt.ylabel("employment length", labelpad=14)
plt.xlabel("Counts of Fully Paid", labelpad=14)
plt.title("loan_status,Fully Paid VS employment length", y=1.02)
ax_1.set_xscale('log')


# In[114]:


##  loan_status VS emp_length

##  lets plot grph for  Charged Off
df_d14_tmp = df_cols2[df_cols2['loan_status'] =='Charged Off']
df_g14_2 = df_d14_tmp[['emp_length']].value_counts()
ax_2 = df_g14_2.plot(kind='barh')
plt.ylabel("employment length", labelpad=14)
plt.xlabel("Counts of Fully Paid", labelpad=14)
plt.title("loan_status,Fully Paid VS employment length", y=1.02)
ax_2.set_xscale('log')

