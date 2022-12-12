#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# %%
df=pd.read_csv("bank-full.csv", sep=";")
# %%

np.random.seed(123)
print(df.head())
print(df.columns)
print(len(df.columns))
# %%

print(df.shape)
print(df.dtypes)

#%%
# EDA on balance,housing,loan,contact,day,month

# count plot on target variable 
sns.countplot(df["y"])
plt.title("Term Deposit Subscription")
plt.show()

# is there any relationship b/w duration of the call and the contact type? 
import plotly.express as px
fig = px.box(df, x="contact", y="duration", color="y")
fig.show()
# conclusion - it can be seen that there is not significant difference between the call duration and the kind of contact we use for communication.
# However, when compared with the target variable, we see that the customers who spend more time on the call with the marketing team tend to have a higher success rate of term deposit subscription


# is there any relationship between the target variable and the contact communication type
sns.countplot(data=df, x="y", hue="contact")
plt.show()
# conclusion - there are more users using cellular devices. Hence, to make sure that the customer can be reached promptly, we can call up on their cellular devices


# During which months do we have a higher acceptance count for term deposit subscription
pd.value_counts(df.day[df.y=="yes"]).plot.bar()
# Observation - Most high chance of subscription when customer is called during the middle of the month (between 12th and 17th date)



# %%
# Features preprocessing 
import copy
df_anushka = copy.deepcopy(df)

#%%
# numerical
# Standard Scaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

df_anushka[["age", "balance"]] =scaler.fit_transform(df_anushka[['age', 'balance']])
df_anushka[["day", "duration"]] =scaler.fit_transform(df_anushka[['day', 'duration']])
df_anushka[["campaign", "pdays"]] =scaler.fit_transform(df_anushka[['campaign', 'pdays']])
df_anushka[["previous"]] =scaler.fit_transform(df_anushka[['previous']])

df_anushka.head()


#%%
# categorical
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
df_anushka['job']= label_encoder.fit_transform(df_anushka["job"]) 
df_anushka['marital']= label_encoder.fit_transform(df_anushka["marital"]) 
df_anushka['education']= label_encoder.fit_transform(df_anushka["education"]) 
df_anushka['contact']= label_encoder.fit_transform(df_anushka["contact"]) 
df_anushka['month']= label_encoder.fit_transform(df_anushka["month"])
df_anushka['poutcome']= label_encoder.fit_transform(df_anushka["poutcome"])

print(df_anushka.head())


#%%
# convert boolean into numerical
df_anushka.default = df_anushka.default.map(dict(yes=1, no=0))
df_anushka.housing = df_anushka.housing.map(dict(yes=1, no=0))
df_anushka.loan = df_anushka.loan.map(dict(yes=1, no=0))
df_anushka.y = df_anushka.y.map(dict(yes=1, no=0))

df_anushka.head()
#%%
# split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

X, y = df_anushka.iloc[:, :-1], df_anushka.iloc[:, -1]
# Train/test set generation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=123
)


#%%
# Base model
from sklearn.linear_model import LogisticRegression 
logisticRegr = LogisticRegression()
_ = logisticRegr.fit(X_train, y_train)
logisticRegr.score(X_test, y_test)

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
logisticRegr = LogisticRegression()
_ = logisticRegr.fit(X_train.loc[:, rfe.support_], y_train)
logisticRegr.score(X_test.loc[:,['age', 'default', 'balance', 'housing', 'loan', 'contact', 'duration',
       'campaign', 'pdays', 'previous']], y_test)


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

import statsmodels.api as sm 
from statsmodels.formula.api import glm
from sklearn.model_selection import train_test_split

X, y = df_anushka.iloc[:, :-1], df_anushka.iloc[:, -1]
# Train/test set generation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=123
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

modelLogit = glm(formula='y ~ C(marital)+C(education)+C(default)+C(housing)+C(loan)+C(contact)+C(month)+duration+campaign+C(poutcome)', data=os_data, family=sm.families.Binomial())

modelLogitFit = modelLogit.fit()
print( modelLogitFit.summary() )

modelpredicitons = pd.DataFrame( columns=['Predicted'], data= modelLogitFit.predict(pd.concat([X_test, y_test], axis=1))) 
cut_off = 0.3
# Compute class predictions
modelpredicitons['Predictions'] = np.where(modelpredicitons['Predicted'] > cut_off, 1, 0)
print(modelpredicitons.Predictions.head())
#
# Make a cross table
confusionmatrix = (pd.crosstab(os_test.y, modelpredicitons.Predictions,
rownames=['Actual'], colnames=['Predicted'],
margins = True))

print(pd.crosstab(os_test.y, modelpredicitons.Predictions,
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

# %%
# Cross fold validations
from sklearn.model_selection import cross_val_score

lr_cv_acc = cross_val_score(logisticRegr, os_data_X, os_data_y, cv= 5, scoring='accuracy' )
print(f'LR CV accuracy score:  {lr_cv_acc}')

