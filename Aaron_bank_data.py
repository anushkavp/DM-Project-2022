#%%
# Package Load Ins
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
#%%
# Import Data
df = pd.read_csv('bank-full.csv', sep=";")
df.head()
import copy
df_model= copy.deepcopy(df)
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
# Chi-Squared Tests on Categorical Variables Against Y Target
#
job_crosstab = pd.crosstab(df['y'],
                            df['job'], 
                               margins = False)
print(job_crosstab)
job_chisq = stats.chi2_contingency(job_crosstab)
print("P-value for job: ", job_chisq[1])
if job_chisq[1] < 0.05:
    print("Significant")
else:
    print("Not Significant")
print()
# %%
marital_crosstab = pd.crosstab(df['y'],
                            df['marital'], 
                               margins = False)
print(marital_crosstab)
marital_chisq = stats.chi2_contingency(marital_crosstab)
print("P-value for marital: ", marital_chisq[1])
if marital_chisq[1] < 0.05:
    print("Significant")
else:
    print("Not Significant")
print()
# %%
education_crosstab = pd.crosstab(df['y'],
                            df['education'], 
                               margins = False)
print(education_crosstab)
education_chisq = stats.chi2_contingency(education_crosstab)
print("P-value for education: ", education_chisq[1])
if education_chisq[1] < 0.05:
    print("Significant")
else:
    print("Not Significant")
print()
# %%
default_crosstab = pd.crosstab(df['y'],
                            df['default'], 
                               margins = False)
print(default_crosstab)
default_chisq = stats.chi2_contingency(default_crosstab)
print("P-value for default: ", default_chisq[1])
if default_chisq[1] < 0.05:
    print("Significant")
else:
    print("Not Significant")
print()
# %%
housing_crosstab = pd.crosstab(df['y'],
                            df['housing'], 
                               margins = False)
print(housing_crosstab)
housing_chisq = stats.chi2_contingency(housing_crosstab)
print("P-value for housing: ", housing_chisq[1])
if housing_chisq[1] < 0.05:
    print("Significant")
else:
    print("Not Significant")
print()
# %%
loan_crosstab = pd.crosstab(df['y'],
                            df['loan'], 
                               margins = False)
print(loan_crosstab)
loan_chisq = stats.chi2_contingency(loan_crosstab)
print("P-value for loan: ", loan_chisq[1])
if loan_chisq[1] < 0.05:
    print("Significant")
else:
    print("Not Significant")
print()
# %%
contact_crosstab = pd.crosstab(df['y'],
                            df['contact'], 
                               margins = False)
print(contact_crosstab)
contact_chisq = stats.chi2_contingency(contact_crosstab)
print("P-value for contact: ", contact_chisq[1])
if contact_chisq[1] < 0.05:
    print("Significant")
else:
    print("Not Significant")
print()
# %%
month_crosstab = pd.crosstab(df['y'],
                            df['month'], 
                               margins = False)
print(month_crosstab)
month_chisq = stats.chi2_contingency(month_crosstab)
print("P-value for month: ", month_chisq[1])
if month_chisq[1] < 0.05:
    print("Significant")
else:
    print("Not Significant")
print()
# %%
poutcome_crosstab = pd.crosstab(df['y'],
                            df['poutcome'], 
                               margins = False)
print(poutcome_crosstab)
poutcome_chisq = stats.chi2_contingency(poutcome_crosstab)
print("P-value for poutcome: ", poutcome_chisq[1])
if poutcome_chisq[1] < 0.05:
    print("Significant")
else:
    print("Not Significant")
print()

#%%
# Preprocess
#
df_model.default = df_model.default.map(dict(yes=1, no=0))
df_model.housing = df_model.housing.map(dict(yes=1, no=0))
df_model.loan = df_model.loan.map(dict(yes=1, no=0))
df_model.y = df_model.y.map(dict(yes=1, no=0))

from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
df_model['job']= label_encoder.fit_transform(df_model["job"]) 
df_model['marital']= label_encoder.fit_transform(df_model["marital"]) 
df_model['education']= label_encoder.fit_transform(df_model["education"]) 
df_model['contact']= label_encoder.fit_transform(df_model["contact"]) 
df_model['month']= label_encoder.fit_transform(df_model["month"])
df_model['poutcome']= label_encoder.fit_transform(df_model["poutcome"])

x_model = df_model.drop(columns=['y'])
y_model = df_model.iloc[:, -1]

#%%
# Training and Test Split
#
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_model,y_model,test_size = 0.3,random_state=10)

# %%
# K-Nearest Neighbor - Full Model (All Variables)
#
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
knn_full = KNeighborsClassifier(n_neighbors=9) 
knn_full.fit(x_train,y_train)
ytest_pred = knn_full.predict(x_test)
# Score report
print(classification_report(y_test,ytest_pred))
print(confusion_matrix(y_test,ytest_pred))

# %%
# KNN -- Reduced Variable Model
# ['age', 'default', 'balance', 'housing', 'loan', 'contact', 'duration','campaign', 'pdays', 'previous']
x_model_red = x_model.copy()
x_model_red = x_model_red.drop(columns=['job', 'marital', 'day', 'poutcome'])
x_train_red, x_test_red, y_train_red, y_test_red = train_test_split(x_model_red,y_model,test_size = 0.3,random_state=10)

knn_red = KNeighborsClassifier(n_neighbors=9) 
knn_red.fit(x_train_red, y_train_red)
ytest_pred_red = knn_red.predict(x_test_red)
# Score report
print(classification_report(y_test_red, ytest_pred_red))
print(confusion_matrix(y_test_red, ytest_pred_red))

# %%
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=123)
x_train_smote, y_train_smote=smote.fit_resample(x_train, y_train)

knn_smote_full = KNeighborsClassifier(n_neighbors=9) 
knn_smote_full.fit(x_train_smote, y_train_smote)
ytest_pred_smote = knn_smote_full.predict(x_test)
# Score report
print(classification_report(y_test, ytest_pred_smote))
print(confusion_matrix(y_test, ytest_pred_smote))

# %%
