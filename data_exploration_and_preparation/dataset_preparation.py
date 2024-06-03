# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Dataset preparation
# #### Authors
# - Giménez López, Antonio
# - Izquierdo Alvarez, Mario
# - Ortega Pinto, Valentina Isabel de la Milagrosa
# - Romero Romero, Martín

# %% [markdown]
# # Introduction to the Problem
#
# Flotation is defined as a physicochemical surface tension process that separates sulfide minerals from other minerals and species that make up the bulk of the original rock. During this process, the ground mineral adheres superficially to previously blown air bubbles, which determines the separation of the mineral of interest.
#
# The adhesion of the mineral to these air bubbles will depend on the hydrophilic properties of each mineral species to be separated from those that have no commercial value, which are called gangue.
#
# The flotation process involves several stages:
#
# * Grinding: the ore is crushed to a size suitable for the liberation of the minerals of interest.
#
# * Conditioning: The ground ore is mixed with water and chemical reagents (e.g. pH modifiers).
#
# * Flotation: In this stage, air is introduced into the conditioned mineral mixture, generating air bubbles that adhere to the valuable mineral particles. These bubbles with attached minerals form a froth on the surface, which is collected as concentrate.
#
# * Cleaning: The concentrate obtained in the primary flotation stage may go through additional flotation stages to improve its grade and purity.
#
# * Thickening and filtration: The final concentrate is thickened and filtered to remove excess water.
#
# Our dataset and therefore our project will focus on trying to **reduce** the amount of **silica** that **remains** in the **ore** at the **end** of the **flotation step.**

# %% [markdown]
# ## Import the needed libraries

# %%
import sys
import os
utils_path = os.path.abspath('../utils/')
if utils_path not in sys.path:
    sys.path.append(utils_path)

# %%
import pandas as pd
from rich import print
import matplotlib.pyplot as plt
import seaborn as sns
from data_visualization import * 

# %% [markdown]
# ## Loading and preparing the dataset
#
# #### Dataset Features 
#
# This dataset contains information on the flotation processes in a mining plant
#
# The dataset contains the following features: (Note that some features are measured each 20 seconds and others each hour)
#
# - **Date**: Date and time of the measurement in *yyyy-mm-dd hh:mm:ss* format.
# - **% Iron Feed**: Percentage of the iron contained in the iron ore which has been introduced into the flotation cells. Changes each hour.
# - **% Silica Feed**: Percentage of the silica (impority) contained in the iron ore which has been introduced into the flotation cells. Changes each hour.
# - **Starch Flow**: Starch (Reagent) flow measured in m^3/h. Measured each 20s.
# - **Amina Flow**: Amina (Reagent) flow measured in m^3/h. Measured each 20s.
# - **Ore Pulp Flow**: Ore flow measured in m³/h. Measured each 20s.
# - **Ore Pulp pH**: Ore pH, scale from 0 to 14. Measured each 20s.
# - **Ore Pulp Density**: Ore density, scale from 1 to 3 kg/cm³. Measured each 20s.
# - **Flotation Air Flow**: Air flow that goes into the flotation cell measured in Nm³/h. Measured each 20s. (One for each of the seven collumns)
# - **Flotation Level**: Froth level in the flotation cell measured in mm (millimeters). Measured each 20s. (One for each of the seven collumns)
# - **% Iron Concentrate**: Percentage of iron which represents how much iron is in the ore in the end of the flotation process. It is a lab measurement and it changes each hour.
# - **% Silica Concentrate**: Percentage of silica which represents how much silica is still presented in the end of the flotation process. It is a lab measurement and it changes each hour.
#
# #### Preprocessing
# Some modifications could be made to the **Mining Process Flotation Plant Database** in order to improve its usability. All the numeric values are actually strings, and the decimal separator is a coma. For a better workflow, a new .csv is going to be generated, replacing comas with dots. Additionally, all numeric data will be parsed into float.

# %%
# Loading data in a pandas DataFrame
csv_file_path = '../data/MiningProcess_Flotation_Plant_Database.csv'

df = pd.read_csv(csv_file_path)

# %%
cols_to_modify = df.columns.drop(['date']) # Only this column is not numeric (not modified)

# %%
for column in cols_to_modify:
    df[column] = df[column].str.replace(',', '.').astype(float)
df

# %% [markdown]
# ### Save the preprocessed dataset

# %%
#save the data
# df.to_csv('data/prepared_miningProcess_data.csv', index=False)

# %% [markdown]
# ## Data exploration
#
# In this section it will be studied the integrity of the dataset, analyzing the percentage of nulls present in each feature. At the same time, a description of each of the variables will be shown to study how they are distributed.
#
# It will be also displayed graphically how the distribution of Silica Concentrate percentage evolves by means of an interactive histogram.

# %%
null_percentaje = df.isnull().mean() * 100

df_summary = df.describe(include='all')
print(f"######## - Null percentage - ########\n {null_percentaje}")
print("\n")
print("######################## - Dataset Description - ########################")
print(df_summary)


# %% [markdown]
# It has been found that the dataset has **no null values** in any of its characteristics, highlighting the **integrity** of the data.
#
# Analyzing the dataset description it is observed that '% Iron Feed', '% Silica Feed', 'Starch Flow', 'Amine Flow', 'Ore Pulp Flow', 'Ore Pulp pH' and 'Ore Pulp Density' present a **different distribution** among them.
#
# On the other hand, for the features related to the columns of the flotation process, it is visualized how for both **Air Flow** and **Level**, the **seven columns** present a **similar distributions**. Going further, a **small difference** can be determined between columns **one, two and three** and columns **four, five, six and seven**.
#
#

# %% [markdown]
# #### Silica Concentrate Percentage over time

# %%
concentrations = df['% Silica Concentrate']
plot_interactive_hist(concentrations)

# %% [markdown]
# During the first samples we have that the distribution is mostly concentrated between **1 and 2 percent**. As more samples are added, a **large peak** is observed around **3.26 percent**, which grradually **smooths** out. Finally, we observe again that the interval between **1 and 2 percent** is the most populated.

# %% [markdown]
# ## Correlation Matix Analysis
#
# The correlation matrix of the dataset is displayed below to show the correlation between each of the features. 

# %%
df_without_date = df.drop(['date'], axis=1)

corr_matrix = df_without_date.corr().abs() # Some are inversely correlated (% iron concentrate)

# Plot heatMap
plt.figure(figsize=(40, 20))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=.5)
plt.title('Heat Map Correlation Matrix')
plt.show()


# %% [markdown]
# After a quick examination, it can be seen that there is a **higher correlation** both in air flow and level between columns **one, two and three** and between columns **four, five, six and seven**; as already observed in the description of the dataset.
#
# Looking at the characteristics with the **highest correlation** with the variable to predict, '**% silica concentrate**', the highest correlation was found to be with the '**% Iron concentrate**' in an **inverse** way (**0.8**). This makes sense given that the **lower** the concentration of **impurities** present in the ore, the **higher** the concentration of **iron** will be in the **final ore**. 
#
# It is important to note that this feature cannot be counted since it is obtained by means of a sample **measured in the laboratory** at the **same time** as '**% Silica concentrate**', so the **impurity concentration** would already be **known** at the time in a **real production escenario**. 

# %% [markdown]
# ## Correlation Matrix without % Iron Concentrate
#
# Here will be display the correlation of each feature with the variable to be inferred, '**% Silica concentrate**' excluding '**% Iron Concentrate**'.

# %%
df_without_IronCon = df.drop(['date', '% Iron Concentrate' ], axis=1)

corr_matrix = df_without_IronCon.corr().abs()

# Lets plot only correlation with the class, without considering itself for better localization of the most correlated features
target_corr = corr_matrix.iloc[:-1, -1]

# Creating a df for using seaborn
target_corr_df = target_corr.to_frame().transpose()

plt.figure(figsize=(30, 20))
sns.heatmap(target_corr_df, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=.5, cbar_kws={"orientation": "horizontal"})
plt.title('Correlation with the Last Column')
plt.show()


# %% [markdown]
#
# Excluding the '**% Iron concentrate**' of the correlation matrix reveals that the features more correlated with '**% Silica concentrate**' are '**Flotation Collumn 01 Air Flow**', '**Flotation Collumn 02 Air Flow**', '**Flotation Collumn 03 Air Flow**' and '**Flotation Collumn 05 Level**' with values of correlation above **0.17 %**. This is consistent with what it was found earlier that there were high correlation and distribution similarity bewteen collumns one, two and three and also between collumns four, five, six and seven.
#
#
# Due to inconsistency in the sampling time of some features, it has been decided to aggregate all features by hour, so a study of the hourly aggregated dataset will be carried out below. A more detailed explanation can be found at `online_learning_experimentation.py`. 

# %% [markdown]
# # Dataset aggregated by hours

# %%
df_agg = df.groupby(df.columns[0]).mean()
df_agg.reset_index(inplace=True)

df_agg

# %% [markdown]
# ## Correlation Matrix without % Iron Concentrate aggregated by hours

# %%
df_without_IronCon_agg = df_agg.drop(['date', '% Iron Concentrate' ], axis=1)

corr_matrix = df_without_IronCon_agg.corr().abs()

# Lets plot only correlation with the class, without considering itself for better localization of the most correlated features
target_corr = corr_matrix.iloc[:-1, -1]

# Creating a df for using seaborn
target_corr_df = target_corr.to_frame().transpose()

plt.figure(figsize=(30, 20))
sns.heatmap(target_corr_df, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=.5, cbar_kws={"orientation": "horizontal"})
plt.title('Correlation with the Last Column')
plt.show()

# %% [markdown]
# The new results are quite similar with the previous results without aggregating, it is observed that '**% Silica concentrate**' are '**Flotation Collumn 01 Air Flow**', '**Flotation Collumn 03 Air Flow**' and '**Flotation Collumn 05 Level**' with values of correlation above **19 %**. These **three features** are the ones that have been **selected** to be **used** in the prediction **models** for both batch and online learning. Given its high correlation and distribution similarity, it has been decided that it would be adecuate to **take into account** the  '**Flotation Collumn 02 Air Flow**' since Air FLow Collums 1 and 3 has been selected as high correlated and also '**Flotation Collumn 04 Level**' and '**Flotation Collumn 07 Level**' as Level Collumn 05 has been selected and they have an aceptable grade of correlation with '**% Silica concentrate**'.
#
# The **rest of features** will be **excluded** in the designed prediction models.

# %% [markdown]
# ### Save the aggregated dataset

# %%
# df_agg.to_csv('data/prepared_miningProcess_data_agg.csv', index=False)
