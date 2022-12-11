#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# %%
df=pd.read_csv("training_data.csv")
# %%

np.random.seed(123)
print(df.head())
print(df.columns)
print(len(df.columns))

#%%
# EDA
# Safety features

# Is parking camera?
import plotly.express as px
fig = px.histogram(df, x="is_parking_camera", color='is_claim')
fig.show()

# is parking sensors?
import plotly.express as px
fig = px.histogram(df, x="is_parking_sensors", color='is_claim')
fig.show()

# is_adjusttable_steering
import plotly.express as px
fig = px.histogram(df, x="is_adjustable_steering", color='is_claim')
fig.show()



# Description of car
#	Age of car 
import plotly.express as px
fig = px.box(df,y="age_of_car", x="is_claim")
fig.show()


#	Model


#	Segment



#%%
# Numerical Variables

# Corrplot

# computing the correlation plot
corr = df.corr()
plt.subplots(figsize=(20,15))
print(corr)
sns.heatmap(corr)

#%%
# PREPROCESSING

## categorical variables into numerical using label encoding

import copy
df_anushka = copy.deepcopy(df)
from sklearn import preprocessing
#%%
label_encoder = preprocessing.LabelEncoder()
df_anushka['area_cluster']= label_encoder.fit_transform(df_anushka["area_cluster"]) 
df_anushka['make']= label_encoder.fit_transform(df_anushka["make"]) 
df_anushka['segment']= label_encoder.fit_transform(df_anushka["segment"]) 
df_anushka['model']= label_encoder.fit_transform(df_anushka["model"]) 
df_anushka['fuel_type']= label_encoder.fit_transform(df_anushka["fuel_type"])
df_anushka['max_torque']= label_encoder.fit_transform(df_anushka["max_torque"])
df_anushka['max_power']= label_encoder.fit_transform(df_anushka["max_power"])
df_anushka['engine_type']= label_encoder.fit_transform(df_anushka["engine_type"])
df_anushka['airbags']= label_encoder.fit_transform(df_anushka["airbags"])
df_anushka['rear_brakes_type']= label_encoder.fit_transform(df_anushka["rear_brakes_type"])
df_anushka['cylinder']= label_encoder.fit_transform(df_anushka["cylinder"])
df_anushka['transmission_type']= label_encoder.fit_transform(df_anushka["transmission_type"])
df_anushka['gear_box']= label_encoder.fit_transform(df_anushka["gear_box"])
df_anushka['steering_type']= label_encoder.fit_transform(df_anushka["steering_type"])
df_anushka['ncap_rating']= label_encoder.fit_transform(df_anushka["ncap_rating"])

print(df_anushka.head())
#%%

# convert boolean into numerical
df_anushka.is_esc = df_anushka.is_esc.map(dict(Yes=1, No=0))
df_anushka.is_adjustable_steering = df_anushka.is_adjustable_steering.map(dict(Yes=1, No=0))
df_anushka.is_tpms = df_anushka.is_tpms.map(dict(Yes=1, No=0))
df_anushka.is_parking_sensors = df_anushka.is_parking_sensors.map(dict(Yes=1, No=0))
df_anushka.is_parking_camera = df_anushka.is_parking_camera.map(dict(Yes=1, No=0))
df_anushka.is_front_fog_lights = df_anushka.is_front_fog_lights.map(dict(Yes=1, No=0))
df_anushka.is_rear_window_wiper = df_anushka.is_rear_window_wiper.map(dict(Yes=1, No=0))
df_anushka.is_rear_window_washer = df_anushka.is_rear_window_washer.map(dict(Yes=1, No=0))
df_anushka.is_rear_window_defogger = df_anushka.is_rear_window_defogger.map(dict(Yes=1, No=0))
df_anushka.is_brake_assist = df_anushka.is_brake_assist.map(dict(Yes=1, No=0))
df_anushka.is_power_door_locks = df_anushka.is_power_door_locks.map(dict(Yes=1, No=0))
df_anushka.is_central_locking = df_anushka.is_central_locking.map(dict(Yes=1, No=0))
df_anushka.is_power_steering = df_anushka.is_power_steering.map(dict(Yes=1, No=0))
df_anushka.is_driver_seat_height_adjustable = df_anushka.is_driver_seat_height_adjustable.map(dict(Yes=1, No=0))
df_anushka.is_day_night_rear_view_mirror = df_anushka.is_day_night_rear_view_mirror.map(dict(Yes=1, No=0))
df_anushka.is_ecw = df_anushka.is_ecw.map(dict(Yes=1, No=0))
df_anushka.is_speed_alert = df_anushka.is_speed_alert.map(dict(Yes=1, No=0))

df_anushka.head()


#%%
# Scaling numerical variables
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

df_anushka[["policy_tenure", "age_of_car"]] =scaler.fit_transform(df_anushka[['policy_tenure', 'age_of_car']])
df_anushka[["age_of_policyholder", "population_density"]] =scaler.fit_transform(df_anushka[['age_of_policyholder', 'population_density']])
df_anushka[["displacement", "turning_radius"]] =scaler.fit_transform(df_anushka[['displacement', 'turning_radius']])
df_anushka[["length", "width"]] =scaler.fit_transform(df_anushka[['length', 'width']])
df_anushka[["height", "gross_weight"]] =scaler.fit_transform(df_anushka[['height', 'gross_weight']])

df_anushka.head()


# %%

# %%

# One Hot Encoding for categorical vars
# from sklearn.preprocessing import OneHotEncoder
# onehotencoder = OneHotEncoder(categories="auto")

# transformed_data = onehotencoder.fit_transform(df_anushka[["area_cluster","make","segment", "model","fuel_type","airbags", "rear_brakes_type","cylinder", "transmission_type","gear_box","steering_type"]]).toarray()
# feature_labels = onehotencoder.get_feature_names()
# # the above transformed_data is an array so convert it to dataframe
# feature_labels = np.array(feature_labels).ravel()
# features = pd.DataFrame(transformed_data, columns=feature_labels)

# # now concatenate the original data and the encoded data using pandas
# concatenated_data = pd.concat([df_anushka, features], axis=1)
# concatenated_data = concatenated_data.drop(["policy_id","area_cluster","make","segment", "model","fuel_type","airbags", "rear_brakes_type","cylinder", "transmission_type","gear_box","steering_type"], axis=1)
# concatenated_data['max_torque']= label_encoder.fit_transform(concatenated_data["max_torque"])
# concatenated_data['max_power']= label_encoder.fit_transform(concatenated_data["max_power"])
# concatenated_data['engine_type']= label_encoder.fit_transform(concatenated_data["engine_type"])

# %%

# Feature selection using featurewiz

## base RandomClassifier Model
### Train and test split
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score 
# %pip install featurewiz
#from featurewiz import featurewiz
np.random.seed(1234)


# %%

# Feature selection model #1
x = df_anushka.iloc[:,1:-1]
y = df_anushka.iloc[:,-1]

#%% 
# with RFE

from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression

# Feature extraction
model = LogisticRegression()
rfe = RFE(model, n_features_to_select=10)
fit = rfe.fit(x,y)
print("Num Features: %s" % (fit.n_features_))
#print("Selected Features: %s" % (fit.support_))
print("Feature Ranking: %s" % (fit.ranking_))
for feature in fit.support_.nonzero():
  print(x.columns[feature])
  
# 'policy_tenure', 'age_of_car', 'age_of_policyholder', 'population_density', "area_cluster", 'fuel_type', 'is_parking_sensors','is_parking_camera', 'rear_brakes_type', 'transmission_type', 'steering_type  

#%%

# Final preprocessing before model making
# Taking only those 10 variables from above
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categories="auto")
x = df[["policy_tenure", "age_of_car", "age_of_policyholder", "population_density", "fuel_type", "is_parking_sensors", "is_parking_camera", "rear_brakes_type","transmission_type","steering_type"]]
y= df[["is_claim"]]

# numerical features scaling
x[["policy_tenure", "age_of_car", "age_of_policyholder", "population_density"]] =scaler.fit_transform(x[["policy_tenure", "age_of_car", "age_of_policyholder", "population_density"]])

# boolean features:
x.is_parking_sensors = x.is_parking_sensors.map(dict(Yes=1, No=0))
x.is_parking_camera = x.is_parking_camera.map(dict(Yes=1, No=0))

# Categorical features 
transformed_data = onehotencoder.fit_transform(x[["fuel_type","rear_brakes_type","transmission_type","steering_type"]]).toarray()
feature_labels = onehotencoder.get_feature_names()
# the above transformed_data is an array so convert it to dataframe
feature_labels = np.array(feature_labels).ravel()
features = pd.DataFrame(transformed_data, columns=feature_labels)

# now concatenate the original data and the encoded data using pandas
concatenated_data = pd.concat([x, features], axis=1)
concatenated_data = concatenated_data.drop(["fuel_type","rear_brakes_type","transmission_type","steering_type"], axis=1)


#%%
# Logistic Regression
# SMOTE
x_train, x_test, y_train, y_test = train_test_split(concatenated_data, y, test_size=0.3, random_state=42)
from imblearn.over_sampling import SMOTE
os = SMOTE(random_state=0)
columns = x_train.columns

os_data_X,os_data_y=os.fit_resample(x_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y= pd.DataFrame(data=os_data_y)

logisticRegr = LogisticRegression()
logisticRegr.fit(os_data_X, os_data_y)

print('Logit model accuracy (with the test set):', logisticRegr.score(x_test, y_test))
print('Logit model accuracy (with the train set):', logisticRegr.score(os_data_X, os_data_y))

print(logisticRegr.predict(x_test))

test = logisticRegr.predict_proba(x_test)
type(test)

from sklearn.metrics import classification_report
y_true, y_pred = y_test, logisticRegr.predict(x_test)
print(classification_report(y_true, y_pred))


from sklearn.metrics import roc_auc_score, roc_curve

# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(y_test))]
# predict probabilities
lr_probs = logisticRegr.predict_proba(x_test)
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
