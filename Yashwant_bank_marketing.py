#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler
import copy

pd.set_option('display.max_columns', None)


#%%
df = pd.read_csv('bank-full.csv', delimiter = ';')
df.rename({'y':'is_subscribed'},axis = 1, inplace = True)
df.head()


# %%
#taking limited columns only as we divided the features for EDA within our group members
EDA_Columns = df.iloc[:,11:]
EDA_Columns.head()


# %%
subscription_taken = df[df['is_subscribed']=='yes']
subscription_not_taken = df[df['is_subscribed']=='no']


# %%
sns.boxplot(data = df,x = 'is_subscribed', y = 'duration')


#%%
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
pie_values
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
print('#campaign : number of contacts performed during this campaign and for this client (numeric, includes last contact)')
plt.figure(figsize = (20,10))
sns.countplot(x = df['campaign'],hue = df['is_subscribed'],data = df)
plt.ylabel('customer count')

print('Most of the customers are contacted once and we can see the value of the frequency of calls till 5. Very few customers are contacted more that 5 to 6 times')


# %%

print('pdays: number of days that passed by after the client was last contacted from a previous campaign')
fig,ax = plt.subplots(1,2,figsize = (10,5))
ax[0].violinplot(subscription_taken['pdays'])
ax[0].title.set_text('subsciption taken')
ax[0].set(ylabel = 'number of days from previous call')
ax[1].violinplot(subscription_not_taken['pdays'])
ax[1].title.set_text('subsciption not taken')
ax[1].set(ylabel = 'number of days from previous call')


# %%
print('Previous call feature: number of contacts performed before this campaign and for this client (numeric)')
fig,ax = plt.subplots(1,2,figsize = (10,5))
ax[0].violinplot(subscription_taken['previous'])
ax[0].title.set_text('subsciption taken')
ax[0].set(ylabel = 'number of previous call')
ax[1].violinplot(subscription_not_taken['previous'])
ax[1].title.set_text('subsciption not taken')
ax[1].set(ylabel = 'number of previous call')



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


std_scl = StandardScaler()
new_df = copy.deepcopy(df)
new_df['age_Scaled'] = std_scl.fit_transform(df['age'].values.reshape(-1,1))
new_df['balance_Scaled'] = std_scl.fit_transform(df['balance'].values.reshape(-1,1))
new_df['duration_scaled'] = std_scl.fit_transform(df['duration'].values.reshape(-1,1))
new_df['pdays_scaled'] = std_scl.fit_transform(df['pdays'].values.reshape(-1,1))
new_df['previous_scaled'] = std_scl.fit_transform(df['previous'].values.reshape(-1,1))
new_df['day_scaled'] = std_scl.fit_transform(df['day'].values.reshape(-1,1))
new_df['campaign_scaled'] = std_scl.fit_transform(df['campaign'].values.reshape(-1,1))
new_df.head()
#%%
from sklearn.preprocessing import LabelEncoder
encode = LabelEncoder()
new_df['month_encoded'] = encode.fit_transform(df['month'])
new_df['job_encoded'] = encode.fit_transform(df['job'])
new_df['marital_encoded'] = encode.fit_transform(df['marital'])
new_df['education_encoded'] = encode.fit_transform(df['education'])
new_df['contact_encoded'] = encode.fit_transform(df['contact'])
new_df['poutcome_encoded'] = encode.fit_transform(df['poutcome'])


#%%
new_df['default_encoded'] = df['default'].map({'no':0,'yes':1})
new_df['housing_encoded'] = df['housing'].map({'no':0,'yes':1})
new_df['loan_encoded'] = df['loan'].map({'no':0,'yes':1})
new_df['is_subscribed_label'] = df['is_subscribed'].map({'no':0,'yes':1})


best_feature_df = new_df[['age_Scaled', 'default_encoded', 'balance_Scaled', 'housing_encoded', 'loan_encoded', 'contact_encoded', 'duration_scaled',
       'campaign_scaled', 'pdays_scaled', 'previous_scaled','is_subscribed_label']]


best_feature_df.head()



####Data splitting

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(best_feature_df.drop(columns = ['is_subscribed_label'], axis = 1),best_feature_df['is_subscribed_label'],test_size = 0.20)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

x_test.head()

#%%
from imblearn.over_sampling import SMOTE

oversample = SMOTE(random_state=123)
X_train_processed_smote, Y_train_processed_smote = oversample.fit_resample(x_train, y_train)

X_train_processed_smote,X_cv_processed_smote,Y_train_processed_smote,\
Y_cv_processed_smote = train_test_split(X_train_processed_smote,Y_train_processed_smote,test_size = 0.2)


# %%
from sklearn.naive_bayes import BernoulliNB
from sklearn.calibration import CalibratedClassifierCV


#####################BernoulliNB#######################

alpha = [0.00001,0.0005, 0.0001,0.005,0.001,0.05,0.01,0.1,0.5,1,5,10,50,100]
cv_recall_score = []

for i in alpha:
    model = BernoulliNB(alpha = i)
    model.fit(X_train_processed_smote,Y_train_processed_smote)
    CV = CalibratedClassifierCV(model,method = 'sigmoid')
    CV.fit(X_train_processed_smote,Y_train_processed_smote)
    predicted = CV.predict(X_cv_processed_smote)
    cv_recall_score.append(recall_score(Y_cv_processed_smote,predicted))
for i in range(0,len(cv_recall_score)):
    print('Recall value for apha =' + str(alpha[i]) + ' is ' + str(cv_recall_score[i]))
plt.plot(alpha,cv_recall_score,c='r')
plt.xlabel('alpha(n_estimators)')
plt.ylabel('recall score')
plt.title('alpha vs recall_score')


#%%
for i,score in enumerate(cv_recall_score):
    plt.annotate((alpha[i],np.round(score,4)),(alpha[i],np.round(cv_recall_score[i],4)))

index = cv_recall_score.index(max(cv_recall_score))
best_alpha = alpha[index]
print('best alpha is ' + str(best_alpha))
model = BernoulliNB(alpha = best_alpha)
model.fit(X_train_processed_smote,Y_train_processed_smote)
predict_train = model.predict(X_train_processed_smote)
print('recall score on train data ' + str(recall_score(Y_train_processed_smote,predict_train)))
train_mat = confusion_matrix(Y_train_processed_smote,predict_train)
predict_cv = model.predict(X_cv_processed_smote)
print('recall score on test data ' + str(recall_score(Y_cv_processed_smote,predict_cv)))
cv_mat = confusion_matrix(Y_cv_processed_smote,predict_cv)
predict_test = model.predict(x_test)
print('recall score on test data ' + str(recall_score(y_test,predict_test)))
test_mat = confusion_matrix(y_test,predict_test)
fig,ax = plt.subplots(1,3,figsize = (15,5))
sns.heatmap(ax = ax[0],data = train_mat,annot=True,fmt='g',cmap="YlGnBu")
ax[0].set_xlabel('predicted')
ax[0].set_ylabel('actual')
ax[0].title.set_text('confusion matrix for train data')
sns.heatmap(ax = ax[1],data = cv_mat,annot=True,fmt='g')
ax[1].set_xlabel('predicted')
ax[1].set_ylabel('actual')
ax[1].title.set_text('confusion matrix for CV data')
sns.heatmap(ax = ax[2],data = test_mat,annot=True,fmt='g')
ax[2].set_xlabel('predicted')
ax[2].set_ylabel('actual')
ax[2].title.set_text('confusion matrix for test data')


#%%
#Random forest updated with gridsearch 

from sklearn.model_selection import GridSearchCV
params = [{'n_estimators' : [10,20,50,100,200,400,500],
         'max_depth': [1,5,10,20,30,50,100]}]
rf = RandomForestClassifier()
rf_gs = GridSearchCV(rf,param_grid=params,
                      scoring='recall',
                      cv=5)
rf_gs.fit(X_train_processed_smote, Y_train_processed_smote)
rf_gs.best_params_


# %%
model = RandomForestClassifier(n_estimators = 100,max_depth = 8)
model.fit(X_train_processed_smote,Y_train_processed_smote)
predict_train = model.predict(x_train)
print('recall score on train data ' + str(recall_score(y_train,predict_train)))
train_mat = confusion_matrix(y_train,predict_train)
predict_test = model.predict(x_test)
print('recall score on test data ' + str(recall_score(y_test,predict_test)))
test_mat = confusion_matrix(y_test,predict_test)
fig,ax = plt.subplots(1,2,figsize = (15,5))
sns.heatmap(ax = ax[0],data = train_mat,annot=True,fmt='g',cmap="YlGnBu")
ax[0].set_xlabel('predicted')
ax[0].set_ylabel('actual')
ax[0].title.set_text('confusion matrix for train data')
sns.heatmap(ax = ax[1],data = test_mat,annot=True,fmt='g')
ax[1].set_xlabel('predicted')
ax[1].set_ylabel('actual')
ax[1].title.set_text('confusion matrix for test data')



#%%
###########################LGBM############################
from sklearn.model_selection import GridSearchCV
params = [{'n_estimators' : [10,20,50,100,200,400,500],
         'max_depth': [1,5,10,20,30,50,100]}]
LGBM_model = LGBMClassifier()
rf_gs = GridSearchCV(LGBM_model,param_grid=params,
                      scoring='recall',
                      cv=5)
rf_gs.fit(X_train_processed_smote, Y_train_processed_smote)
rf_gs.best_params_


# %%
model = LGBMClassifier(n_estimators = 100,max_depth = 8)
model.fit(X_train_processed_smote,Y_train_processed_smote)
predict_train = model.predict(x_train)
print('recall score on train data ' + str(recall_score(y_train,predict_train)))
train_mat = confusion_matrix(y_train,predict_train)
predict_test = model.predict(x_test)
print('recall score on test data ' + str(recall_score(y_test,predict_test)))
test_mat = confusion_matrix(y_test,predict_test)
fig,ax = plt.subplots(1,2,figsize = (15,5))
sns.heatmap(ax = ax[0],data = train_mat,annot=True,fmt='g',cmap="YlGnBu")
ax[0].set_xlabel('predicted')
ax[0].set_ylabel('actual')
ax[0].title.set_text('confusion matrix for train data')
sns.heatmap(ax = ax[1],data = test_mat,annot=True,fmt='g')
ax[1].set_xlabel('predicted')
ax[1].set_ylabel('actual')
ax[1].title.set_text('confusion matrix for test data')
