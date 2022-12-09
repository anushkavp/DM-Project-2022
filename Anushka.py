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
# Numerical Variables

# Corrplot

# computing the correlation plot
corr = df.corr()
plt.subplots(figsize=(20,15))
print(corr)
sns.heatmap(corr)

#%%
# categorical variables into numerical






# %%

# Feature selection using featurewiz
