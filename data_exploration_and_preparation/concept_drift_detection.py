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
# # Concept Drift dectector - Mining Process Flotation Plant Database
#
# #### Authors:
# - Giménez López, Antonio
# - Izquierdo Alvarez, Mario
# - Ortega Pinto, Valentina Isabel de la Milagrosa
# - Romero Romero, Martín
#
#
# ## Introduction
#
# In this notebook, we will execute various concept drift detectors to verify the accuracy of our initial hypothesis. We will utilize the aggregated dataset for this purpose, as it is the dataset selected for training both batch and online learning models. It's important to note that the online learning method will employ the non-aggregated original dataset. However, the averages will be calculated dynamically, and only the data instances from the aggregated dataset will be used for training purposes.

# %% [markdown]
# ## Importing the needed packages
# First, is needed to add `utils/` to the syspath to use the custom functions

# %%
import sys
import os
utils_path = os.path.abspath('../utils/')
if utils_path not in sys.path:
    sys.path.append(utils_path)

# %%
from data_visualization import * 

# %%
from river import drift
import pandas as pd
from rich import print
import matplotlib.pyplot as plt

from tqdm import tqdm

# %% [markdown]
# ## Loading prepared Data
# This data has been taken from [Kaggle](https://www.kaggle.com/datasets/edumagalhaes/quality-prediction-in-a-mining-process) and has been previously **preprocessed** by aggregating by date and computing the **means** of the columns.

# %%
# Loading data in a pandas DataFrame
csv_file_path = '../data/prepared_miningProcess_data_agg.csv'

df = pd.read_csv(csv_file_path)

# %%
# Sorting data by date
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S')

df = df.sort_values(by='date')
df

# %% [markdown]
# The concept drift detectors to be used, check for distribution changes on the data features. Thus, some columns need to be dropped: **date**, **% Iron Concentrate** and **% Silica Concentrate**

# %%
# Drop the date and targets
features_to_monitor = df.columns.drop(['date', '% Iron Concentrate', '% Silica Concentrate'])
features_to_monitor

# %% [markdown]
# ## Using Adaptive Windowing method for concept drift detection
#
# Adaptive Windowing method for concept drift detection.
#
# ADWIN (ADaptive WINdowing) is a popular drift detection method with mathematical guarantees. ADWIN efficiently keeps a variable-length window of recent items; such that it holds that there has no been change in the data distribution. This window is further divided into two sub-windows 
# used to determine if a change has happened. ADWIN compares the average of 
# and 
# to confirm that they correspond to the same distribution. Concept drift is detected if the distribution equality no longer holds. Upon detecting a drift, 
# is replaced by 
# and a new 
# is initialized. ADWIN uses a significance value 
# to determine if the two sub-windows correspond to the same distribution.
#
# Explanation taken from [River Documentation](https://riverml.xyz/dev/api/drift/ADWIN/)

# %%
# Creating an ADWIN detector for each feature
detectors = {feature: drift.ADWIN() for feature in features_to_monitor}
changes = {feature: 0 for feature in features_to_monitor}
drifts_dict = {feature: [] for feature in features_to_monitor}
for index, row in df.iterrows():
    # Update the each deterctor and check for concept drift
    for feature in features_to_monitor:
        value = row[feature]
        detector = detectors[feature]
        detector.update(value)
        
        if detector.drift_detected:
            changes[feature] += 1
            drifts_dict[feature].append(index)


# %%
for feature, drifts in changes.items():
    print(f"For feature {feature}, {drifts} drifts have been detected")

# %%
feature =  'Flotation Column 04 Level'
plot_concept_drift(df[feature].values, drifts_dict[feature])

# %% [markdown]
# ### Considerations
# As can be clearly seen, **several concept drifts** have been detected in **each feature** by the **ADWIN** detector. These results **support** our concept drift **initial hypothesis**. However, we are going to use another concept drift detector to verify these results.

# %% [markdown]
# ## Using Kolmogorov-Smirnov Windowing method for concept drift detection.
#
#
# KSWIN (Kolmogorov-Smirnov Windowing) is a concept change detection method based on the Kolmogorov-Smirnov (KS) statistical test. The KS-test is a non-parametric test that does not assume any specific underlying data distribution. KSWIN is capable of monitoring data or performance distributions and requires the input to be one-dimensional arrays.
#
# KSWIN operates by maintaining a sliding window of a fixed size (window_size). It considers the last stat_size samples within this window to represent the current concept. From the first portion of the window, a number of samples are uniformly drawn, representing an approximation of the current concept.
#
# The KS-test is then applied to two windows of the same size. This test measures the distance between the empirical cumulative distributions of these windows.
#
# A concept drift is detected by KSWIN if the difference in empirical data distributions between the two windows is significant, indicating that they do not originate from the same distribution.
#
# Explanation taken from [River Documentation](https://riverml.xyz/dev/api/drift/KSWIN/)

# %%
# Create a KSWIN detector per feature
detectors = {feature: drift.KSWIN() for feature in features_to_monitor}
changes = {feature: 0 for feature in features_to_monitor}
drifts_dict = {feature: [] for feature in features_to_monitor}

# Using tqdm for visulizing a progress bar.
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    for feature in features_to_monitor:
        value = row[feature]
        detector = detectors[feature]
        detector.update(value)
        
        if detector.drift_detected:
            changes[feature] += 1
            drifts_dict[feature].append(index)


# %%
for feature, drifts in changes.items():
    print(f"For feature {feature}, {drifts} drifts have been detected")

# %%
feature =  'Flotation Column 04 Level'
plot_concept_drift(df[feature].values, drifts_dict[feature])

# %% [markdown]
# ### Considerations
# Once again, the above results show **several concept drift detections** at **each feature**, **supporting** both our **initial hypothesis** and the **ADWIN results**. Given these results, we can assume that **concept drift is present on our dataset**, thus, it can take **advantage** of **online learning capabilities**.

# %% [markdown]
# ## Interactive histogram to visualize the evolution of '% Silica Concentrate' distribution

# %% [markdown]
# For visualization purposes, we have plotted the below **interactive** histogram. It allows seeing how the distribution of `% Silica Concentrate` **evolves**.

# %%
concentrations = df['% Silica Concentrate']
plot_interactive_hist(concentrations, figsize=(20,10))

# %%

# %%
