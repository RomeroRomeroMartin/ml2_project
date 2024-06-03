# %% [markdown]
# # Online vs Batch Learningn Comparison
#
# In this notebook, different figures will be shown comparing the performance of the different models used both in batch and online learning to compare the error results and to establish whether there has been a significant reduction in that prediction error when using online learning techniques versus batch learning techniques.  

# %% [markdown]
# ### Authors
#
# * Romero Romero, Martin
# * Izquierdo Alvarez, Mario
# * Ortega Pinto, Valentina
# * Giménez López, Antonio

# %% [markdown]
# ## Import the needed libraries

# %%
import os
import sys

utils_path = os.path.abspath('../utils/')
if utils_path not in sys.path:
    sys.path.append(utils_path)

# %%
import pickle
import pandas as pd
from rich import print
from data_visualization import * 

# %% [markdown]
# ## Load the metrics

# %%
# Using pickle and pandas to load the metrics

# On-line learning metrics
with open('../online_learning/OL_results.pkl', 'rb') as f:
    OL_data = pickle.load(f)
# Batch learning metrics
BL_data = pd.read_csv('../batch_learning/batch_results.csv')
BL_metrics = dict(zip(BL_data.iloc[:, 0], BL_data.iloc[:, 1]))


# %% [markdown]
# ## Preprocessing
#
# For the comparison between both Batch and Online learning, in order to be consistent, it will be **only considered** the **instances** corresponding to the **test dataset** used in the **batch learning**, given the metrics for the **batch** learning algorithms has been obtained from a **test dataset**, which represent the **33%** of the whole dataset. In this manner, those instances will be isolated.
#
# It is important to note that the MAE metrics for the online learning algorithms have been computed for the entire data stream, so these metrics will have a drag on the error computed on data outside the batch learning test data set. 
#
# This is not a big problem for this comparison, since it allows for a pessimistic evaluation and if it turns out that online learning provides a better return for this problem, it can be sure of it.

# %%
# Taking only the instances corresponding to the test dataset used in the batch learning algorithm

OL_metrics = {}
for keys, values in OL_data.items():
    OL_metrics[keys] = values[-1353:]

# %% [markdown]
# ## Comparison
#
# These linear plots will show the **performance** of each of the **models** developed using online learning **along the test data stream** with the **average performance** of its **specific model** developed using **batch learning**, which will be a **constant**.
#
# This is to **test** whether the use of **online** learning techniques has **improved** the **performance** in predicting the **Silica Concentration** compared to the predictions of the **batch** learning model.

# %%
plot_online_batch_comparison(OL_metrics['Adaptative Random Forest'],algorithm='Adaptative Random Forest Regressor',constant_line=BL_metrics['MAE RFRegressor'], y_lim=[0.65 , 0.95], figsize=(10,5))
plot_online_batch_comparison(OL_metrics['K-Nearest Neighbors'],algorithm='K-Nearest Neighbors Regressor',constant_line=BL_metrics['MAE KnnRegressor'], y_lim=[0.65 , 0.95], figsize=(10,5))
plot_online_batch_comparison(OL_metrics['Hoeffding Adaptative Tree Regressor'],algorithm='Hoeffding Adaptative Tree Regressor',constant_line=BL_metrics['MAE DTRegressor'], y_lim=[0.65 , 0.95], figsize=(10,5))

# %% [markdown]
# ## Results discussion and performance comparison
#
# Analysing the three plots, it can be seen that the three **online learning** models have a fairly **stable MAE** for the test samples. In addition, it can be seen that both the **Hoeffding Tree Regressor** and the **Adaptive Random Forest Regressor** have a **lower level of error** over all the instances in comparison with **K-Nearest Neighbors Regressor**, as was the case in the online learning results (`online_learning_experimentation.py`).
#
# On the other hand, it can be seen that the **MAE** of all the models developed in the **online learning** is **lower** than the **average MAE** of the corresponding model in the **batch learning**. This suggests that the **online learning approach** to **this problem** could bring a **performance improvement** over traditional approaches like **batch learning**. Therefore, the use of **online learning** is shown to provide **better predictive capability** in silica concentration, with which engineers could **readjust the flotation process** to **reduce the percentage of silica concentration** in each separation process.
#
#
