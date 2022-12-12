#%%
# Package Load Ins
import pandas as pd
import numpy as np
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

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

df_model[["age", "balance"]] =scaler.fit_transform(df_model[['age', 'balance']])
df_model[["day", "duration"]] =scaler.fit_transform(df_model[['day', 'duration']])
df_model[["campaign", "pdays"]] =scaler.fit_transform(df_model[['campaign', 'pdays']])
df_model[["previous"]] =scaler.fit_transform(df_model[['previous']])

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
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

for k in (3,5,7,9,11,13,15):
    knn_full = KNeighborsClassifier(n_neighbors=k) 
    knn_full.fit(x_train,y_train)
    ytest_pred = knn_full.predict(x_test)
    # Score report
    print(k)
    print(classification_report(y_test,ytest_pred))
    print(confusion_matrix(y_test,ytest_pred))
    print()

# %%
# KNN -- Reduced Variable Model
# ['age', 'default', 'balance', 'housing', 'loan', 'contact', 'duration','campaign', 'pdays', 'previous']
x_model_red = x_model.copy()
x_model_red = x_model_red.drop(columns=['job', 'marital', 'day', 'poutcome'])
x_train_red, x_test_red, y_train_red, y_test_red = train_test_split(x_model_red,y_model,test_size = 0.3,random_state=10)

for k in (3,5,7,9,11,13,15):
    knn_red = KNeighborsClassifier(n_neighbors=k) 
    knn_red.fit(x_train_red, y_train_red)
    ytest_pred_red = knn_red.predict(x_test_red)
    # Score report
    print(k)
    print(classification_report(y_test_red, ytest_pred_red))
    print(confusion_matrix(y_test_red, ytest_pred_red))
    print()

# %%
# KNN -- Smote Enhanced Full Model 
#
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=123)
x_train_smote, y_train_smote=smote.fit_resample(x_train, y_train)
for k in (3,5,7,9,11,13,15):
    knn_smote_full = KNeighborsClassifier(n_neighbors=k) 
    knn_smote_full.fit(x_train_smote, y_train_smote)
    ytest_pred_smote = knn_smote_full.predict(x_test)
    # Score report
    print(k)
    print(classification_report(y_test, ytest_pred_smote))
    print(confusion_matrix(y_test, ytest_pred_smote))
    print()

# %%
# KNN -- Smote Enhanced Reduced Variable Model 
#
smote_red = SMOTE(random_state=123)
x_train_smote_red, y_train_smote_red = smote_red.fit_resample(x_train_red, y_train_red)
for k in (3,5,7,9,11,13,15):
    knn_smote_red = KNeighborsClassifier(n_neighbors=k) 
    knn_smote_red.fit(x_train_smote_red, y_train_smote_red)
    ytest_pred_smote_red = knn_smote_red.predict(x_test_red)
    # Score report
    print(k)
    print(classification_report(y_test_red, ytest_pred_smote_red))
    print(confusion_matrix(y_test_red, ytest_pred_smote_red))
    print()


#%%
# from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
# knn_cvs = KNeighborsClassifier(n_neighbors=15)
# scoring = {'accuracy' : make_scorer(accuracy_score), 
#            'precision' : make_scorer(precision_score),
#            'recall' : make_scorer(recall_score), 
#            'f1_score' : make_scorer(f1_score)}
# xmodel_result = cross_validate(knn_cvs, x_train_smote, y_train_smote, cv=10, scoring=scoring)
# knn_cvs.fit(x_train_smote, y_train_smote)
# knn_cvs.score(x_train_smote, y_train_smote)
# print("Accuracy: ",xmodel_result['test_accuracy'].mean())
# print("Precision: ",xmodel_result['test_precision'].mean())
# print("Recall: ",xmodel_result['test_recall'].mean())
# print("F1 Score: ",xmodel_result['test_f1_score'].mean())


# %%
knn_cvs = KNeighborsClassifier(n_neighbors=13)
xmodel_result = cross_val_score(knn_cvs, x_train_smote, y_train_smote, cv=10)
knn_cvs.fit(x_train_smote, y_train_smote)
knn_cvs.score(x_train_smote, y_train_smote)

#%%
#
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

knn_smote_3 = KNeighborsClassifier(n_neighbors=13) 
knn_smote_3.fit(x_train_smote_red, y_train_smote_red)
ytest_pred_3 = knn_smote_3.predict(x_test_red)
fpr, tpr, threshold = roc_curve(y_test, ytest_pred_3)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curve of kNN')
plt.show()
# %%
