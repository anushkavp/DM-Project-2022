#%%
# Package Load Ins
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import pyplot
# from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import copy
#%%
# Import Data
df = pd.read_csv('bank-full.csv', sep=";")
df.head()
# %%
# EDA - Age, Job, Marital, Education, Default, Housing, Loan
#
# Fixing catgorical variables
df.iloc[:,16:17] = df.iloc[:,16:17].astype("category")
df.iloc[:,1:5] = df.iloc[:,1:5].astype("category")
df.iloc[:,1:5] = df.iloc[:,1:5].astype("category")
df.iloc[:,6:8] = df.iloc[:,6:8].astype("category")
df.iloc[:,6:8] = df.iloc[:,6:8].astype("category")
df.iloc[:,6:8] = df.iloc[:,6:8].astype("category")
df['contact'] = df['contact'].astype("category")
df['month'] = df['month'].astype("category")
df['poutcome'] = df['poutcome'].astype("category")
#%% 
# Age Variable - All Ages
sns.set_style("whitegrid")
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.boxplot(x=df["age"], data=df, color='red').set(title='Client Ages', xlabel="Age")

#%% 
# Ages Based on Y Target
sns.set_style("whitegrid")
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.boxplot(x=df["y"], y=df["age"],data=df).set(title='Client Ages based on Binary', ylabel="Age", xlabel="")

#%%
# Job Variable - Count 
sns.set_style("whitegrid")
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.countplot(y = df["job"], data=df).set(title='Jobs Worked by Clients', xlabel="Count", ylabel="Jobs")

#%%
# Job Variable - Plot with Y Target Variable
sns.set_style("whitegrid")
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.countplot(y=df["job"], hue=df["y"]).set(title='Jobs Worked by Clients', xlabel="Count", ylabel="Jobs")

#%%
# Marital Variable
sns.set_style("whitegrid")
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.countplot(x = df["marital"], data=df).set(title='Marital Status by Clients', xlabel="Marital Status", ylabel="Count")

#%%
# Marital Variable - Plot with Y Target Variable
sns.set_style("whitegrid")
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.countplot(x=df["marital"], hue=df["y"]).set(title='Marital Status by Clients based on Binary', xlabel="Marital Status", ylabel="Count")

#%%
# Education Variable
sns.set_style("whitegrid")
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.countplot(x = df["education"], data=df).set(title='Education Level', xlabel="Marital Status", ylabel="Count")

#%%
# Education Variable - Plot with Y Target Variable
sns.set_style("whitegrid")
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.countplot(x=df["education"], hue=df["y"]).set(title='Education by Clients based on Binary', xlabel="Education Level", ylabel="Count")

#%%
# Default Variable
sns.set_style("whitegrid")
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.countplot(x = df["default"], data=df).set(title='Default', xlabel="", ylabel="Count")

#%%
# Default Variable - Plot with Y Target Variable
sns.set_style("whitegrid")
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.countplot(x=df["default"], hue=df["y"]).set(title='Default based on Binary', xlabel="", ylabel="Count")

#%% 
# Balance Variable - All Ages
sns.set_style("whitegrid")
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.boxplot(x=df["balance"], data=df, color='red').set(title='Client Balances', xlabel="Balance")
#%% 
# Balance Based on Y Target
sns.set_style("whitegrid")
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.boxplot(x=df["y"], y=df["balance"],data=df).set(title='Client Balance based on Binary', ylabel="Balance", xlabel="")

#%%
# Balance Plotted Against Ages
sns.set_style("whitegrid")
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=df, x=df["age"], y=df["balance"]).set(title='Client Balance based on Binary', ylabel="Balance", xlabel="Age")

#%%
# Balance Plotted Against Ages Based on Y Target
sns.set_style("whitegrid")
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=df, x=df["age"], y=df["balance"],hue=df["y"])

#%%
# Housing Variable
sns.set_style("whitegrid")
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.countplot(x = df["housing"], data=df).set(title='Housing', xlabel="", ylabel="Count")

#%%
# Housing Variable - Plot with Y Target Variable
sns.set_style("whitegrid")
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.countplot(x=df["housing"], hue=df["y"]).set(title='Housing based on Binary', xlabel="", ylabel="Count")

#%%
# Loan Variable
sns.set_style("whitegrid")
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.countplot(x = df["housing"], data=df).set(title='Loan', xlabel="", ylabel="Count")

#%%
# Loan Variable - Plot with Y Target Variable
sns.set_style("whitegrid")
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.countplot(x=df["loan"], hue=df["y"]).set(title='Loan based on Binary', xlabel="", ylabel="Count")

#%% 
# Target Variable Countplot
sns.set_style("whitegrid")
sns.countplot(x = df["y"], data=df).set(title='Client Subscribed to a Term Deposit', xlabel="", ylabel="Count")

responses = [df[df['y']=='no'].shape[0], df[df['y']=='yes'].shape[0]]
labels= ["No", "Yes"]
plt.figure(facecolor="white")
plt.pie(responses,labels=labels, autopct='%2.1f%%',
        shadow=False, startangle=180)
plt.title("Bank Term Deposit Substription Percentages")
plt.show()

# %%
df.rename({'y':'is_subscribed'},axis = 1, inplace = True) #renaming target column
subscription_taken = df[df['is_subscribed']=='yes']
subscription_not_taken = df[df['is_subscribed']=='no']



#%%
#Violin plot for call duration based on term deposit subscription
fig,ax = plt.subplots(1,2,figsize = (10,5))
sns.violinplot(ax = ax[0],x = subscription_taken['duration'])
ax[0].title.set_text('subsciption taken')
ax[0].set(xlabel='duration(in Seconds)')
sns.violinplot(ax = ax[1],x = subscription_not_taken['duration'])
ax[1].title.set_text('subsciption not taken')
ax[1].set(xlabel='duration(in Seconds)')


# %%
#####################((Poutcome)previous Campaign outcome)###################
print('Poutcome')
pie_values = df['poutcome'].value_counts().reset_index().rename(columns={'index':'poutcome','poutcome':'count'})
pie_values #get count for all outcome categories
plt.pie(pie_values['count'],labels=pie_values['poutcome'],autopct='%.0f%%')
plt.show()

print('For most of the customers, campaign status is unknown. Success rate of campaign is very low but there might me the chance that most of the customers which are the part of successful campaign are in unknown category.')

# %%
########################month and days############################## 
print('Month and day info')
# 
df['day'].value_counts() 

# %%
df['month'].value_counts()

# %%
jan = df[df['month']=='jan']
feb = df[df['month']=='feb']
mar = df[df['month']=='mar']
apr = df[df['month']=='apr']
may = df[df['month']=='may']
jun = df[df['month']=='jun']
jul = df[df['month']=='jul']
aug = df[df['month']=='aug']
sep = df[df['month']=='sep']
oct = df[df['month']=='oct']
nov = df[df['month']=='nov']
dec = df[df['month']=='dec']


# %%
fig,(axis) = plt.subplots(3,4,figsize = (20,20))
axis[0,0].bar(list(jan['day'].value_counts().index),jan['day'].value_counts().
values,color = 'orange')
axis[0,0].title.set_text('daywise Jan phonecalls')
axis[0,0].set_xlabel('days of month')
axis[0,0].set_ylabel('phone call count')
axis[0,0].set_ylim(0,1000)
axis[0,1].bar(list(feb['day'].value_counts().index),feb['day'].value_counts().
values,color = 'green')
axis[0,1].title.set_text('daywise feb phonecalls')
axis[0,1].set_xlabel('days of month')
axis[0,1].set_ylim(0,1000)
axis[0,2].bar(list(mar['day'].value_counts().index),mar['day'].value_counts().
values,color = 'red')
axis[0,2].title.set_text('daywise mar phonecalls')
axis[0,2].set_xlabel('days of month')
axis[0,2].set_ylim(0,1000)
axis[0,3].bar(list(apr['day'].value_counts().index),apr['day'].value_counts().
values,color = 'blue')
axis[0,3].title.set_text('daywise apr phonecalls')
axis[0,3].set_xlabel('days of month')
axis[0,3].set_ylim(0,1000)
axis[1,0].bar(list(may['day'].value_counts().index),may['day'].value_counts().
values,color = 'orange')
axis[1,0].title.set_text('daywise may phonecalls')
axis[1,0].set_xlabel('days of month')
axis[1,0].set_ylim(0,1000)
axis[1,0].set_ylabel('phone call count')
axis[1,1].bar(list(jun['day'].value_counts().index),jun['day'].value_counts().
values,color = 'green')
axis[1,1].title.set_text('daywise Jun phonecalls')
axis[1,1].set_xlabel('days of month')
axis[1,1].set_ylim(0,1000)
axis[1,2].bar(list(jul['day'].value_counts().index),jul['day'].value_counts().
values,color = 'red')
axis[1,2].title.set_text('daywise Jul phonecalls')
axis[1,2].set_xlabel('days of month')
axis[1,2].set_ylim(0,1000)
axis[1,3].bar(list(aug['day'].value_counts().index),aug['day'].value_counts().
values,color = 'blue')
axis[1,3].title.set_text('daywise aug phonecalls')
axis[1,3].set_xlabel('days of month')
axis[1,3].set_ylim(0,1000)
axis[2,0].bar(list(sep['day'].value_counts().index),sep['day'].value_counts().
values,color = 'orange')
axis[2,0].title.set_text('daywise sep phonecalls')
axis[2,0].set_xlabel('days of month')
axis[2,0].set_ylim(0,1000)
axis[0,0].set_ylabel('phone call count')
axis[2,1].bar(list(oct['day'].value_counts().index),oct['day'].value_counts().
values,color = 'green')
axis[2,1].title.set_text('daywise oct phonecalls')
axis[2,1].set_xlabel('days of month')
axis[2,1].set_ylim(0,1000)
axis[2,2].bar(list(nov['day'].value_counts().index),nov['day'].value_counts().
values,color = 'red')
axis[2,2].title.set_text('daywise nov phonecalls')
axis[2,2].set_xlabel('days of month')
axis[2,2].set_ylim(0,1000)
axis[2,3].bar(list(dec['day'].value_counts().index),dec['day'].value_counts().
values,color = 'blue')
axis[2,3].title.set_text('daywise dec phonecalls')
axis[2,3].set_xlabel('days of month')
axis[2,3].set_ylim(0,1000)


# %%
##########################EDA on campaign feature###############################
print('#campaign : number of contacts performed during this campaign and for this client (numeric, includes last contact)')
plt.figure(figsize = (20,10))
sns.countplot(x = df['campaign'],hue = df['is_subscribed'],data = df)
plt.ylabel('customer count')

print('Most of the customers are contacted once and we can see the value of the frequency of calls till 5. Very few customers are contacted more that 5 to 6 times')

# %%
####################EDA on pdays#####################################
print('pdays: number of days that passed by after the client was last contacted from a previous campaign')
fig,ax = plt.subplots(1,2,figsize = (10,5))
ax[0].violinplot(subscription_taken['pdays'])
ax[0].title.set_text('subsciption taken')
ax[0].set(ylabel = 'number of days from previous call')
ax[1].violinplot(subscription_not_taken['pdays'])
ax[1].title.set_text('subsciption not taken')
ax[1].set(ylabel = 'number of days from previous call')


# %%
##############################EDA on 'Previous' feature############################
print('Previous call feature: number of contacts performed before this campaign and for this client (numeric)')
fig,ax = plt.subplots(1,2,figsize = (10,5))
ax[0].violinplot(subscription_taken['previous'])
ax[0].title.set_text('subsciption taken')
ax[0].set(ylabel = 'number of previous call')
ax[1].violinplot(subscription_not_taken['previous'])
ax[1].title.set_text('subsciption not taken')
ax[1].set(ylabel = 'number of previous call')

#As we are not getting the proper distribution, we are checking the percentile values

# %%
#checking the every 5th percentile of the distribution 
li = [i for i in range(0,101,5)] #finding every 5th quantile
import numpy as np
for i in range(0,len(li)):
    print(str(li[i])+'th quantile is'+str(np.percentile(df['previous'],li[i])))

# %%
#As we are getting big numbers in last 5 percentile, checking it in detail
li = [i for i in range(95,101)]
for i in range(0,len(li)):
    print(str(li[i])+'th quantile is'+str(np.percentile(df['previous'],li[i])))


# %%
print('#As we are getting 99 percentile as 8.9, we are replacing all the other values to 9 calls.')
#df['previous'] = df['previous'].apply(lambda x : 9 if x>8 else None)
df['previous'] = np.where(df['previous'] > 8, 9, df['previous'])
subscription_taken['previous'] = np.where(subscription_taken['previous'] > 8, 9, subscription_taken['previous'])
subscription_not_taken['previous'] = np.where(subscription_not_taken['previous'] > 8, 9, subscription_not_taken['previous'])


#%%
fig,ax = plt.subplots(1,2,figsize = (10,5))
ax[0].violinplot(subscription_taken['previous'])
ax[0].title.set_text('subsciption taken')
ax[0].set(ylabel = 'number of previous call')
ax[1].violinplot(subscription_not_taken['previous'])
ax[1].title.set_text('subsciption not taken')
ax[1].set(ylabel = 'number of previous call')



# %%
# Chi-Squared Tests on Categorical Variables Against Y Target
#
import scipy.stats as stats
catName = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
pValue = []

job_crosstab = pd.crosstab(df['is_subscribed'],
                            df['job'], 
                               margins = False)
print(job_crosstab)
job_chisq = stats.chi2_contingency(job_crosstab)
print("P-value for job: ", job_chisq[1])
pValue.append(job_chisq[1])
if job_chisq[1] < 0.05:
    print("Significant")
else:
    print("Not Significant")
print()
# %%
marital_crosstab = pd.crosstab(df['is_subscribed'],
                            df['marital'], 
                               margins = False)
print(marital_crosstab)
marital_chisq = stats.chi2_contingency(marital_crosstab)
print("P-value for marital: ", marital_chisq[1])
pValue.append(marital_chisq[1])
if marital_chisq[1] < 0.05:
    print("Significant")
else:
    print("Not Significant")
print()
# %%
education_crosstab = pd.crosstab(df['is_subscribed'],
                            df['education'], 
                               margins = False)
print(education_crosstab)
education_chisq = stats.chi2_contingency(education_crosstab)
print("P-value for education: ", education_chisq[1])
pValue.append(education_chisq[1])
if education_chisq[1] < 0.05:
    print("Significant")
else:
    print("Not Significant")
print()
# %%
default_crosstab = pd.crosstab(df['is_subscribed'],
                            df['default'], 
                               margins = False)
print(default_crosstab)
default_chisq = stats.chi2_contingency(default_crosstab)
print("P-value for default: ", default_chisq[1])
pValue.append(default_chisq[1])
if default_chisq[1] < 0.05:
    print("Significant")
else:
    print("Not Significant")
print()
# %%
housing_crosstab = pd.crosstab(df['is_subscribed'],
                            df['housing'], 
                               margins = False)
print(housing_crosstab)
housing_chisq = stats.chi2_contingency(housing_crosstab)
print("P-value for housing: ", housing_chisq[1])
pValue.append(housing_chisq[1])
if housing_chisq[1] < 0.05:
    print("Significant")
else:
    print("Not Significant")
print()
# %%
loan_crosstab = pd.crosstab(df['is_subscribed'],
                            df['loan'], 
                               margins = False)
print(loan_crosstab)
loan_chisq = stats.chi2_contingency(loan_crosstab)
print("P-value for loan: ", loan_chisq[1])
pValue.append(loan_chisq[1])
if loan_chisq[1] < 0.05:
    print("Significant")
else:
    print("Not Significant")
print()
# %%
contact_crosstab = pd.crosstab(df['is_subscribed'],
                            df['contact'], 
                               margins = False)
print(contact_crosstab)
contact_chisq = stats.chi2_contingency(contact_crosstab)
print("P-value for contact: ", contact_chisq[1])
pValue.append(contact_chisq[1])
if contact_chisq[1] < 0.05:
    print("Significant")
else:
    print("Not Significant")
print()
# %%
month_crosstab = pd.crosstab(df['is_subscribed'],
                            df['month'], 
                               margins = False)
print(month_crosstab)
month_chisq = stats.chi2_contingency(month_crosstab)
print("P-value for month: ", month_chisq[1])
pValue.append(month_chisq[1])
if month_chisq[1] < 0.05:
    print("Significant")
else:
    print("Not Significant")
print()
# %%
poutcome_crosstab = pd.crosstab(df['is_subscribed'],
                            df['poutcome'], 
                               margins = False)
print(poutcome_crosstab)
poutcome_chisq = stats.chi2_contingency(poutcome_crosstab)
print("P-value for poutcome: ", poutcome_chisq[1])
pValue.append(poutcome_chisq[1])
if poutcome_chisq[1] < 0.05:
    print("Significant")
else:
    print("Not Significant")
print()

# %%
