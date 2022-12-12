#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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

# %%
