#%%
# Package Load Ins
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import pyplot
from lightgbm import LGBMClassifier
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