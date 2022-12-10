
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import rfit 
import scipy as spy
import scipy.stats as stats

# %%
# First 14 variables EDA
testData = pd.read_csv('test_data.csv', index_col= 'policy_id') 
trainedData = pd.read_csv('training_data.csv', index_col= 'policy_id')
# %%
# Very Clean Data
testData.isna().any()
trainedData.isna().any()

test14 = testData.iloc[:, 0:14].copy()
trained14 = trainedData.iloc[:, 0:14].copy()
trained14 = trained14.join(trainedData['is_claim'])
trained14['is_claim'] = trained14['is_claim'].astype("category")
trained14['max_power'] = trained14['max_power'].str[:-11]
trained14['max_power'] = trained14['max_power'].astype(float)
# %%
trained14['max_torque'] = trained14['max_torque'].str[:-10]
trained14['max_torque'] = trained14['max_torque'].astype(float)

# %%
# 
trained14['engine_type'] = trained14['engine_type'].astype("category")
trained14['model'] = trained14['model'].astype("category")
trained14['fuel_type'] = trained14['fuel_type'].astype("category")
trained14['segment'] = trained14['segment'].astype("category")
trained14['airbags'] = trained14['airbags'].astype("category")
trained14['area_cluster'] = trained14['area_cluster'].astype("category")
trained14['is_esc'] = trained14['is_esc'].astype("category")
#%%
trained14['is_claim'] = trained14['is_claim'].replace(to_replace = 0, value = 'No')
trained14['is_claim'] = trained14['is_claim'].replace(to_replace = 1, value = 'Yes')
claimed = trained14[trained14['is_claim'] == 'Yes']
no_claim = trained14[trained14['is_claim'] == 'No']

#%%
# Cleaning columns Max Power and Max Torque

# %%
# Overlapping Points for Torque and Power
sns.scatterplot(x='max_power', y='max_torque', data=trained14)
sns.set_style('whitegrid')
#%%
sns.displot(y='max_power', data=trained14).set(title='Distribution of Max Power')
#%%
sns.displot(y='max_torque', data=trained14).set(title='Distribution of Max Torque')
#%%
sns.barplot(x='model', y='max_power', data=trained14).set(title='Max Power by Model')
sns.set_style('whitegrid')
# %%
sns.barplot(x='model', y='max_torque', data=trained14).set(title='Max Torque by Model')
sns.set_style('whitegrid')
# %%
sns.barplot(x='fuel_type', y='max_power', data=trained14).set(title='Max Power by Fuel Type')
# %%
sns.barplot(x='fuel_type', y='max_torque', data=trained14).set(title='Max Torque by Fuel Type')
# %%
sns.scatterplot(x='max_power', y='max_torque', data=trained14, hue='fuel_type').set(title='Power vs Torque colored by Fuel Type')
# %%
sns.displot(y='area_cluster', data=trained14).set(title='Distribution of Area Cluster')

#%%
areaCluster_crosstab = pd.crosstab(trained14['is_claim'],
                            trained14['area_cluster'], 
                               margins = False)
statAC, pAC, dofAC, expectedAC = stats.chi2_contingency(areaCluster_crosstab)


segment_crosstab = pd.crosstab(trained14['is_claim'],
                            trained14['segment'], 
                               margins = False)
statSEG, pSEG, dofSEG, expectedSEG = stats.chi2_contingency(areaCluster_crosstab)

model_crosstab = pd.crosstab(trained14['is_claim'],
                            trained14['model'], 
                               margins = False)
statSEG, pSEG, dofSEG, expectedSEG = stats.chi2_contingency(areaCluster_crosstab)

fuelType_crosstab = pd.crosstab(trained14['is_claim'],
                            trained14['fuel_type'], 
                               margins = False)
statSEG, pSEG, dofSEG, expectedSEG = stats.chi2_contingency(areaCluster_crosstab)

engineType_crosstab = pd.crosstab(trained14['is_claim'],
                            trained14['engine_type'], 
                               margins = False)
statSEG, pSEG, dof, expected = stats.chi2_contingency(areaCluster_crosstab)

airbags_crosstab = pd.crosstab(trained14['is_claim'],
                            trained14['airbags'], 
                               margins = False)
stat, p, dof, expected = stats.chi2_contingency(areaCluster_crosstab)

is_esc_crosstab = pd.crosstab(trained14['is_claim'],
                            trained14['is_esc'], 
                               margins = False)
stat, p, dof, expected = stats.chi2_contingency(areaCluster_crosstab)
# %%
