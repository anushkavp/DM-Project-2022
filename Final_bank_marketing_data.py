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

#%%
# 
# is there any relationship b/w duration of the call and the contact type? 
import plotly.express as px
fig = px.box(df, x="contact", y="duration", color="is_subscribed")
fig.update_layout(
    title="Duration of call vs Contact Communication",
    xaxis_title="Contact Communication Type",
    yaxis_title="Duration of call (in seconds)",
    legend_title="Subscribed?",
)
fig.show()

#%%
ax=sns.countplot(data=df, x="is_subscribed", hue="contact")
ax.set_title("Contact Type vs Subscription")
ax.set_xlabel('Term Deposit Subscribed?')
ax.set_ylabel('Count of users')
plt.show()


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


#%%
# Standard Scaling
std_scl = StandardScaler()
new_df = copy.deepcopy(df)
new_df['age'] = std_scl.fit_transform(df['age'].values.reshape(-1,1))
new_df['balance'] = std_scl.fit_transform(df['balance'].values.reshape(-1,1))
new_df['duration'] = std_scl.fit_transform(df['duration'].values.reshape(-1,1))
new_df['pdays'] = std_scl.fit_transform(df['pdays'].values.reshape(-1,1))
new_df['previous'] = std_scl.fit_transform(df['previous'].values.reshape(-1,1))
new_df['day'] = std_scl.fit_transform(df['day'].values.reshape(-1,1))
new_df['campaign'] = std_scl.fit_transform(df['campaign'].values.reshape(-1,1))
new_df.head()

# %%
# categorical
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
new_df['job']= label_encoder.fit_transform(new_df["job"]) 
new_df['marital']= label_encoder.fit_transform(new_df["marital"]) 
new_df['education']= label_encoder.fit_transform(new_df["education"]) 
new_df['contact']= label_encoder.fit_transform(new_df["contact"]) 
new_df['month']= label_encoder.fit_transform(new_df["month"])
new_df['poutcome']= label_encoder.fit_transform(new_df["poutcome"])

print(new_df.head())


#%%
# convert boolean into numerical
new_df['default'] = df['default'].map({'no':0,'yes': 1})
new_df['housing'] = df['housing'].map({'no':0,'yes':1})
new_df['loan'] = df['loan'].map({'no':0,'yes':1})
new_df['is_subscribed'] = df['is_subscribed'].map({'no':0,'yes':1})

new_df["default"] = new_df["default"].astype(float)
new_df["housing"] = new_df["housing"].astype(float)
new_df["loan"] = new_df["loan"].astype(float)
new_df["is_subscribed"] = new_df["is_subscribed"].astype(float)
new_df.head()
# %%
X, y = new_df.iloc[:, :-1], new_df.iloc[:, -1]
# Train/test set generation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)


#%%
# Base model
from sklearn.linear_model import LogisticRegression 
logisticRegr1 = LogisticRegression()
_ = logisticRegr1.fit(X_train, y_train)
logisticRegr1.score(X_test, y_test)
from sklearn.metrics import classification_report
y_true, y_pred = y_test, logisticRegr1.predict(X_test)
print(classification_report(y_true, y_pred))

#%% RFE
from sklearn.feature_selection import RFE

# Init the transformer
rfe = RFE(estimator= LogisticRegression(), n_features_to_select=10)

# Fit to the training data
_ = rfe.fit(X_train, y_train)
# took 50 seconds

print("The best 10 columns are =>",X_train.loc[:, rfe.support_].columns)

#%% 
# model with few dimensions
logisticRegr2 = LogisticRegression()
_ = logisticRegr2.fit(X_train.loc[:, rfe.support_], y_train)
logisticRegr2.score(X_test.loc[:,['age', 'default', 'balance', 'housing', 'loan', 'contact', 'duration',
       'campaign', 'pdays', 'previous']], y_test)
from sklearn.metrics import classification_report
y_true, y_pred = y_test, logisticRegr2.predict(X_test.loc[:,['age', 'default', 'balance', 'housing', 'loan', 'contact', 'duration',
       'campaign', 'pdays', 'previous']])
print(classification_report(y_true, y_pred))

#%% smote
X_train = X_train.loc[:,['age', 'default', 'balance', 'housing', 'loan', 'contact', 'duration',
       'campaign', 'pdays', 'previous']]
X_test = X_test.loc[:,['age', 'default', 'balance', 'housing', 'loan', 'contact', 'duration',
       'campaign', 'pdays', 'previous']]

from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
os = SMOTE(random_state=123)
columns = X_train.columns

os_data_X,os_data_y=os.fit_resample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y= pd.DataFrame(data=os_data_y)

logisticRegr = LogisticRegression()
logisticRegr.fit(os_data_X, os_data_y)

print('Logit model accuracy (with the test set):', logisticRegr.score(X_test, y_test))
print('Logit model accuracy (with the train set):', logisticRegr.score(os_data_X, os_data_y))

print(logisticRegr.predict(X_test))

test = logisticRegr.predict_proba(X_test)
type(test)

from sklearn.metrics import classification_report
y_true, y_pred = y_test, logisticRegr.predict(X_test)
print(classification_report(y_true, y_pred))



#%% 
# ROC-AUC score
from sklearn.metrics import roc_auc_score, roc_curve

# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(y_test))]
# predict probabilities
lr_probs = logisticRegr.predict_proba(X_test)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# calculate scores
ns_auc = roc_auc_score(y_test, ns_probs)
lr_auc = roc_auc_score(y_test, lr_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
# plot the roc curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()
# %%
# Statsmodel
import numpy as np
import statsmodels.api as sm 
from statsmodels.formula.api import glm
from sklearn.model_selection import train_test_split

X, y = new_df.iloc[:, :-1], new_df.iloc[:, -1]
# Train/test set generation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)

from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
os = SMOTE(random_state=123)
columns = X_train.columns

os_data_X,os_data_y=os.fit_resample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y= pd.DataFrame(data=os_data_y)

os_data = pd.concat([os_data_X.reset_index(drop=True), os_data_y], axis=1)
os_test = pd.concat([X_test.reset_index(drop=True), y_test], axis=1)

modelLogit = glm(formula='is_subscribed ~ marital+education+default+housing+loan+contact+month+duration+campaign+poutcome', data=os_data, family=sm.families.Binomial())

modelLogitFit = modelLogit.fit()
print( modelLogitFit.summary() )

modelpredicitons = pd.DataFrame( columns=['Predicted'], data= modelLogitFit.predict(pd.concat([X_test, y_test], axis=1))) 
cut_off = 0.3
# Compute class predictions
modelpredicitons['Predictions'] = np.where(modelpredicitons['Predicted'] > cut_off, 1, 0)
print(modelpredicitons.Predictions.head())
#
# Make a cross table
confusionmatrix = (pd.crosstab(os_test.is_subscribed, modelpredicitons.Predictions,
rownames=['Actual'], colnames=['Predicted'],
margins = True))

print(pd.crosstab(os_test.is_subscribed, modelpredicitons.Predictions,
rownames=['Actual'], colnames=['Predicted'],
margins = True))

TN = confusionmatrix.iloc[0,0]
FP = confusionmatrix.iloc[0,1]
FN = confusionmatrix.iloc[1,0]
TP = confusionmatrix.iloc[1,1]
accuracy = (TP+TN)/(TP+TN+FP+FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
specificity = TN / (TN + FP)
print("Accuracy = ", accuracy)
print("Precision = ",precision)
print("Recall/Sensitivity = ",recall)
print("Specificity = ",specificity)
print("F1 score = ",(2*(precision)*(recall))/(precision + recall))
print("AUC score = ",roc_auc_score(y_test, modelLogitFit.predict(pd.concat([X_test, y_test], axis=1))))

# %%
# Cross fold validations
from sklearn.model_selection import cross_val_score

lr_cv_acc = cross_val_score(logisticRegr, os_data_X, os_data_y, cv= 5, scoring='accuracy' )
print(f'LR CV accuracy score:  {lr_cv_acc}')


# SVC
# (took 8 mins to execute)
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report
from sklearn.svm import SVC, LinearSVC
clf1 = LinearSVC()
clf2 = SVC(kernel="linear")
clf3 = SVC()
clf4 = SVC(kernel='rbf', probability=True, C=1, gamma=0.1)
classifiers = [clf1,clf2,clf3, clf4] 

# for c in classifiers:
#     print("Classifier: ",c)
#     c.fit(os_data_X,os_data_y) 
#     print(f'svc train score:  {c.score(X_train,y_train)}')
#     print(f'svc test score:  {c.score(X_test,y_test)}')
#     pred = c.predict(X_test)
#     print(confusion_matrix(y_test, pred))
#     print(classification_report(y_test, pred))
clf1.fit(os_data_X, os_data_y)
print(f'svc train score:  {clf1.score(X_train,y_train)}')
print(f'svc test score:  {clf1.score(X_test,y_test)}')
pred = clf1.predict(X_test)
print(confusion_matrix(y_test, clf1.predict(X_test)))
print(classification_report(y_test, clf1.predict(X_test)))


#%%
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

# Compute fpr, tpr, thresholds and roc auc
fpr, tpr, thresholds = roc_curve(y_test, pred)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate or (1 - Specifity)')
plt.ylabel('True Positive Rate or (Sensitivity)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
# %%

# SVC with different gamma values
# took 17 mins to execute
gammas = ["auto","scale", 0.1,1]
for gamma in gammas:
    print("GAMMA = ", gamma)
    svc = SVC(gamma=gamma).fit(os_data_X, os_data_y)
    print(f'svc train score:  {svc.score(os_data_X, os_data_y)}')
    print(f'svc test score:  {svc.score(X_test,y_test)}')
    pred = svc.predict(X_test)
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred))
    

# %%
