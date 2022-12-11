#%%
import pandas as pd
pd.set_option('display.max_columns', None)


#%%
df = pd.read_csv('train_qWM28Yl.csv')
df.head()

#%%
df['is_claim'].value_counts()


# %%
bool_columns = ['is_esc', 'is_adjustable_steering', 'is_tpms', 'is_parking_sensors', 'is_parking_camera',
               'is_front_fog_lights', 'is_rear_window_wiper', 'is_rear_window_washer', 'is_rear_window_defogger',
               'is_brake_assist', 'is_power_door_locks','is_central_locking','is_power_steering',
                'is_driver_seat_height_adjustable','is_day_night_rear_view_mirror','is_ecw', 'is_speed_alert']

# %%

#converting Yes/No features to 1 and 0.
replace_bool = {'Yes' : 1,  'No': 0}
df[bool_columns] = df[bool_columns].replace(replace_bool)


#%%

ordinal_col = ['max_torque', 'max_power', 'transmission_type', 'steering_type']
dummy_columns = ['area_cluster', 'segment', 'model', 'fuel_type', 'rear_brakes_type', 'engine_type']


#%%

#applying one hot encoding to the features
df = pd.concat([df,pd.get_dummies(df[dummy_columns])],axis=1)
df.drop(dummy_columns, axis=1, inplace=True)


#%%

#converting some of the features in ordinal way

df['transmission_type'] = df['transmission_type'].replace({'Manual' : 1, 'Automatic' : 2})
df['steering_type'] = df['steering_type'].replace({'Manual' : 1, 'Power' : 2, 'Electric': 3})


#Transforming some features from the 
df[['max_torque_Nm', 'max_torque_rpm']] = df["max_torque"].apply(lambda x: pd.Series(str(x).split("@")))
df.drop(["max_torque"], axis=1, inplace= True)
df['max_torque_rpm'] = df['max_torque_rpm'].str[:-3].astype(int)
df['max_torque_Nm'] = df['max_torque_Nm'].str[:-2].astype(float)

df[['max_power_bhp', 'max_power_rpm']] = df["max_power"].apply(lambda x: pd.Series(str(x).split("@")))
df.drop(["max_power"], axis=1, inplace= True)
df['max_power_rpm'] = df['max_power_rpm'].str[:-3].astype(int)
df['max_power_bhp'] = df['max_power_bhp'].str[:-3].astype(float)




#%%
#scaling numerical features
from sklearn.preprocessing import StandardScaler
std_scl = StandardScaler()
df['policy_tenure'] = std_scl.fit_transform(df['policy_tenure'].values.reshape(-1,1))
df['age_of_car'] = std_scl.fit_transform(df['age_of_car'].values.reshape(-1,1))
df['age_of_policyholder'] = std_scl.fit_transform(df['age_of_policyholder'].values.reshape(-1,1))
df['population_density'] = std_scl.fit_transform(df['population_density'].values.reshape(-1,1))
df['displacement'] = std_scl.fit_transform(df['displacement'].values.reshape(-1,1))
df['turning_radius'] = std_scl.fit_transform(df['turning_radius'].values.reshape(-1,1))
df['length'] = std_scl.fit_transform(df['length'].values.reshape(-1,1))
df['width'] = std_scl.fit_transform(df['width'].values.reshape(-1,1))
df['height'] = std_scl.fit_transform(df['height'].values.reshape(-1,1))
df['gross_weight'] = std_scl.fit_transform(df['gross_weight'].values.reshape(-1,1))


df.drop(columns = ['policy_id'], inplace = True, axis = 1)
df.head()


# %%
df.head()



#%%
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(df.drop(columns='is_claim'),df['is_claim'],test_size = 0.20)


#Weight balancing
Yes_datapoints = 54844
No_datapoints =3748
weight = 54844/3748
weight

#weight for balaning the class will be 14.63287.

#%%
from sklearn.ensemble import RandomForestClassifier
feature_names = [f"feature {i}" for i in range(x_train.shape[1])]
forest = RandomForestClassifier(random_state=0,class_weight={0:1,1:14.63287})
forest.fit(x_train, y_train)


#%%
import numpy as np
import matplotlib.pyplot as plt

#%%
importances = forest.feature_importances_
#Sort the feature importance in descending order
sorted_indices = np.argsort(importances)[::-1]
# plt.title('Feature Importance')
# plt.bar(x_train.columns, forest.feature_importances_, align='center')
# plt.xticks(x_train.columns, x_train.columns[sorted_indices], rotation=90)
# plt.tight_layout()
# plt.rcParams["figure.figsize"] = (10,10)
# plt.show()



#%%
x_train.columns



# %%
info  = {'Imp_Features':x_train.columns,'Values':importances}
feature_info = pd.DataFrame(info)
feature_info.head()
# %%
sorted_features = feature_info.sort_values(by='Values', ascending=False).head(40)
# %%
import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})
a = sns.barplot(data = sorted_features, x = "Imp_Features", y = "Values")
a.tick_params(axis='x', rotation=90)



# %%
sorted_features.head(30)

#Some of features like policy_tenure,age_of_policyholder,age_of_car,population_density and area_cluster
#Will make significant impact on it. 


# %%
