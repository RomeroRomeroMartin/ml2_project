# # Online Learning
# #### Authors
# - Giménez López, Antonio
# - Izquierdo Alvarez, Mario
# - Ortega Pinto, Valentina Isabel de la Milagrosa
# - Romero Romero, Martín

# ## Importing needed packages

# First, is needed to add `utils/` to the syspath to use the custom functions

import sys
import os
utils_path = os.path.abspath('../utils/')
if utils_path not in sys.path:
    sys.path.append(utils_path)

from online_learning import build_pipeline, train_online
from data_visualization import * 

# +
import pandas as pd
import matplotlib.pyplot as plt
from statistics import mean
from rich import print
import numpy as np

#river
from river import stream, tree, neighbors, forest
# -

# ## Loading and preparing data

# This data has been taken from [Kaggle](https://www.kaggle.com/datasets/edumagalhaes/quality-prediction-in-a-mining-process)

# Using pandas to load the dataset
df = pd.read_csv('../data/prepared_miningProcess_data.csv')

# **Sorting** data by date to ensure a correct **ordering** on the data stream

# +
# Sorting data by date
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S')

df_sorted = df.sort_values(by='date')
# -

df_sorted

# ### Distribution of Silica Concentrate % over time
# - Use the **slider** to check the data distribution evolution over time

concentrations = df_sorted['% Silica Concentrate']
plot_interactive_hist(concentrations)

# ### Dropping some features
# - The remaining features have been **selected** based on the **correlation** study made on `../dataset_preparation.py`
# - We preserve here the **datetime** feature although it should be removed later on, this is because this feature is needed to know when a **process has ended** and a new process starts (This happens **every hour**)
# - **% Silica Concentrate**, the impurity of the ore at the end of the process (hourly) is the value to predict, based on the exposed requierements at the [Kaggle Dataset](https://www.kaggle.com/datasets/edumagalhaes/quality-prediction-in-a-mining-process)
# - It is noteworthy that **% Iron Concentrate** has been dropped despite it's high correlation with the class to be predicted. This value is almost inversely proportional to **Silica Concentrate** and both are measured at the same time at the end of the process. Using this value as a feature for prediction will lead to great results, however, this **will not** be **useful nor realistic** in a **real production scenario**


def get_stream(data):
    data_stream = stream.iter_pandas(X=data.drop(columns=['% Silica Concentrate', '% Iron Feed', '% Silica Feed', 'Starch Flow', 'Amina Flow', 'Ore Pulp Flow', 'Ore Pulp pH', 
                                                           'Ore Pulp Density', 
                                                           'Flotation Column 04 Air Flow', 'Flotation Column 05 Air Flow', 'Flotation Column 06 Air Flow', 'Flotation Column 07 Air Flow', 
                                                           'Flotation Column 01 Level', 'Flotation Column 02 Level', 'Flotation Column 03 Level', 
                                                           'Flotation Column 06 Level', '% Iron Concentrate']),
                                 y=data['% Silica Concentrate'])
    return data_stream


data_stream = get_stream(df_sorted) 


# ## Some initial preparations

# First, let's print an instance to check that everything is right

x, y = next(iter(data_stream))
x, y

# Now, two lists need to be prepared:
#
# - **features_to_mean:** We want to make a prediction **every 20 seconds**, however, the value to be predicted can only be known after **an hour**, when the process has already **ended**. The most informative prediction can be made at the end of the entire process, aggregating the obtained data across the entire process. Nonetheless, to assists engineers during the process, a prediction is made every 20 seconds using a **mean** of the **already measured** for the **current process**
# - **features_to_discard:** We are **not** really interested on the measurements on a **specific timestamp**, the really **interesting** data is the **aggregation** of the measures since the process has **started**. Thus, after the **mean** of the measures is updated with the current instance inside the pipeline, the data for the specific timestamp is **discarded** before being presented to the model. This means, the **real input** to the model is the **mean** of the features **from the start** of the process **to the current** instant of time.

# +
to_discard = list(x.keys())
features_to_mean = to_discard.copy()

# Now remove features that can't be mean
features_to_mean.remove('date')
features_to_mean, to_discard
# -

# ## Hoeffding Tree Regressor

# ### Building the pipeline
# - **TranformerUnion** with all necessary aggregators
# - **Discard** the selected features
# - **StandardScaler** to standardize the instance
# - **HodeffdingTreeRegressor**
#
# To check how the pipeline is created navigate to `../utils/online_learning.py`

pipeline = build_pipeline(tree.HoeffdingTreeRegressor(max_depth=5, splitter=tree.splitter.TEBSTSplitter(digits=3)), features_to_mean, to_discard)
pipeline

# ### Online Training
#
# Note that as previously said, a prediction is needed **every 20 seconds**, but the real value of the target can only be obtained **after an hour**, when the process is finished. To train and test in a **fair** and **useful** way we should simulate a real scenario, usaging the available resources, so several steps are made:
#
# - For each process (**one hour**), a prediction is made **every 20 seconds** computing the **mean** of **all the available measurements** at the prediction moment, and using this **aggregated vector** as the **input** of the model. However, until the **end** of the process the **real target value can't be obtained**, thus, all these predictions are **collected** to be used later, once the **% Silica Concentrate** is known for that process.
# - **At the end** of the process, the now available real target value is used to **update the metric**. **Each of the predictions made is considered** to compute the metric, thus ensuring a **fair** error measure.
# - The **training** of the model is done only **once per process**, using the **aggregated** measurements **at the end of the process** and the **real target value**. The reason for using only the last instance with the means of all the process for training, and discarding all the intermediate inputs used for the other predictions is the following: In the same process a **quite high** number of predictions are made, but all of them expect **the same** target as a prediction. If **all** these instances are used to train the model with **exactly** the **same target**, the model is likely to **overfit** to that specific target in the **exact same moment** in which the target value changes (due to a new cycle in the process). Thus, the target won't be the same anymore, potentially **leading to bad results**.
#
# > In summary, the **fair** way for dealing with this problem is considering **all the predictions** (every 20 seconds) to measure the error, as all predictions can be used by the engineers, but **only** considering **for training** the **most informed** instance of the proces: **The last** one with the **mean measurements** of the **entire process**. Taking into consideration the later, it is expected that as the process develops, the predictions tend to become **slightly more accurate**, since the inputs to the model are **better informed**.
#
#
#
#
#  
#
# To check how the training function works in detail, it can be checked at  `../utils/online_learning.py`

# Training online
history_HT = train_online(pipeline, data_stream, show_metric=False)

# ### Results visualization for Hoeffding Tree Regressor

plot_metric_history(history_HT)

predictions = [v for v in history_HT['predictions'] if v > 0 and v < 6]

# Drawing of the histogram
plot_histogram(predictions)

# For the next plots, is needed to remove the first instance, as it has not been processed by the model because of `x, y = next(iter(data_stream))` in section **Some initial preparations**

plot_actual_vs_predicted(concentrations[1:], history_HT['predictions'])

# ### Temporal results
#
# The error is **not the only important** measure, in this problem there are **important time constraints**. A new instance could arrive every **20 seconds**, thus, the model should be **always ready** to make a prediction. To ensure this, we have measured the time it takes for the model to make a prediction **since a new instance arrives**. As can be seen in the plot below, despite the high peaks, **the prediction time always remains under 20 seconds with a significant margin**.

plot_time_history(history_HT['times'], algorithm='Hoeffding Tree Regressor')

# ## Adaptive Random Forest

# ### Reset data stream iterator
# We need to create again `data_stream` to iterate again through the stream.

data_stream = get_stream(df_sorted)

# ### Building the pipeline
# - **TranformerUnion** with all necessary aggregators
# - **Discard** the selected features
# - **StandardScaler** to standardize the instance
# - **forest.ARFRegressor**
#
# To check how the pipeline is created navigate to `../utils/online_learning.py`

pipeline = build_pipeline(forest.ARFRegressor(n_models=104, aggregation_method='median',lambda_value=6,max_depth=8,seed=42), features_to_mean, to_discard)
pipeline

# ### Online Training
# To check how the training function works in detail, it can be checked at  `../utils/online_learning.py`

# Training online
history_ARF = train_online(pipeline, data_stream, show_metric=False)

# ### Results visualization for Adaptive Random Forest Regressor

plot_metric_history(history_ARF)

predictions = history_ARF['predictions']

# Dibujando el histograma
plot_histogram(predictions)

plot_actual_vs_predicted(concentrations, history_ARF['predictions'])

# ### Temporal results

plot_time_history(history_ARF['times'], algorithm='Adaptive Random Forest Regressor')

# ## K-Nearest Neighbors regressor

# ### Reset data stream iterator
# We need to create again `data_stream` to iterate again through the stream.

data_stream = get_stream(df_sorted)

# ### Building the pipeline
# - **TranformerUnion** with all necessary aggregators
# - **Discard** the selected features
# - **StandardScaler** to standardize the instance
# - **neighbors.KNNRegressor**
#
# To check how the pipeline is created navigate to `../utils/online_learning.py`

pipeline = build_pipeline(neighbors.KNNRegressor(n_neighbors=17), features_to_mean, to_discard)
pipeline

# ## Online Training
# To check how the training function works in detail, it can be checked at  `../utils/online_learning.py`

# Training online
history_KNN = train_online(pipeline, data_stream, show_metric=False)

# ### Results visualization for K-Nearest Neighbors Regressor

plot_metric_history(history_KNN)

#predictions = [v for v in history_KNN['predictions'] if v > 0 and v < 6]
predictions = history_KNN['predictions']

# Dibujando el histograma
plot_histogram(predictions)

plot_actual_vs_predicted(concentrations, history_KNN['predictions'])

# ### Temporal Results

plot_time_history(history_KNN['times'], algorithm='K-Nearest Neighbors Regressor')

# ## Results comparison
# Now the error results of the different online methods are going to be plotted and compared. In order to avoid the error associated with almost random predictions at early stages of training typically associated with online learning, only the last 70% of the data will be considered.

# +
# Load metric values in a dict to plot the bars
# Only the last 70% of data is considered

start_index = int(len(history_HT['metric_values']) * 0.3)  # It discards the initial 30% of data

data_dict = {
    'Hoeffding Adaptative Tree Regressor': history_HT['metric_values'][start_index:],
    'Adaptative Random Forest': history_ARF['metric_values'][start_index:],
    'K-Nearest Neighbors': history_KNN['metric_values'][start_index:]
}

plot_means_and_stds_with_labels(data_dict)


# +
data_dict = {
    'Hoeffding Adaptative Tree Regressor': history_HT['metric_values'],
    'Adaptative Random Forest': history_ARF['metric_values'],
    'K-Nearest Neighbors': history_KNN['metric_values']
}

plot_many_metrics(data_dict,colors=['seagreen', 'saddlebrown', 'steelblue'],figsize=(20,10), y_lim=(0.7,0.8))
# -

# ## Results discussion and performance comparison
# Upon examining the results with their respective graphs, it can be observed an **initial instability** in the performance of the **Hoeffding tree Regressor**. However, after several iterations, it **rapidly improves** its performance, achieving outcomes comparable to those of the **Adaptive Random Forest Regressor**. The latter, while displaying a **slower learning velocity**, is characterized by its **temporal stability**, consistently offering **superior results** in comparison to other algorithms. On the contrary, the **K-Nearest Neighbors** exhibits a more **erratic behavior**, **struggling** to improve the MAE and even becoming **less precise** over time.
#
# Among the evaluated models, the **Adaptive Random Forest Regressor** demonstrates a more **favorable performance**, maintaining an **optimal balance** between precision and stability. It is noteworthy, however, that the **processing time** for the Random Forest is **significantly longer** than that of the Hoeffding Tree Regressor, as reflected in their respective total execution times (**220.89 vs 17.17 seconds**). Considering the **time constraints of the problem** at hand (one instance every **20 seconds**), **both** algorithms are perfectly **viable**, exhibiting a **substantial margin** between the mean prediction time and the 20-second window available. This renders the **Random Forest Regressor** a quite **suitable** option for this scenario.

# ### Save the results with pickle

# +
import pickle
name = 'OL_results.pkl'
with open(name, 'wb') as file:
    pickle.dump(data_dict, file)

print(f"Online Learning results dictionary saved at '{name}'")

