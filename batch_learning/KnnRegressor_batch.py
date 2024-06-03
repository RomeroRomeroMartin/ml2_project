# %% [markdown]
# # Knn Regressor -- Batch Learning
#
# ### Authors
#
# * Romero Romero, Martin
# * Izquierdo Alvarez, Mario
# * Ortega Pinto, Valentina
# * Giménez López, Antonio

# %% [markdown]
# ## Importing needed packages

# %% [markdown]
# First, is needed to add `utils/` to the syspath to use the custom functions

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
from tqdm import tqdm
import numpy as np
import csv 
from data_visualization import * 

# %%
# Loading data in a pandas DataFrame
csv_file_path = '../data/prepared_miningProcess_data_agg.csv'

df = pd.read_csv(csv_file_path)

# %%
# Sorting data by date
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S')

df = df.sort_values(by='date')
df

# %%
#Separating targets from inputs 
df_inputs = df.columns.drop(['date','% Iron Concentrate','% Silica Concentrate','% Iron Feed', '% Silica Feed', 'Starch Flow', 'Amina Flow', 'Ore Pulp Flow', 'Ore Pulp pH', 
                                                           'Ore Pulp Density', 
                                                           'Flotation Column 04 Air Flow', 'Flotation Column 05 Air Flow', 'Flotation Column 06 Air Flow', 'Flotation Column 07 Air Flow', 
                                                           'Flotation Column 01 Level', 'Flotation Column 02 Level', 'Flotation Column 03 Level', 
                                                           'Flotation Column 06 Level'])
df_targets = df[['% Silica Concentrate']]

# %% [markdown]
# **Thirty-three percent** of the samples were selected as a **test set**, respecting the **time series** of the data, so that the **regression error** calculated on these batch learning data can be **compared** with the error values obtained in the online learning models on the corresponding instances in the data stream.

# %%
# Splitting the dataset into a training set and a test set
train_size = int(len(df)*(1-0.33)) 

x_train, x_test = df[df_inputs].iloc[:train_size], df[df_inputs].iloc[train_size:]
y_train, y_test = df_targets.iloc[:train_size], df_targets.iloc[train_size:]

# %%
x_trained = np.ravel(x_train)
y_trained = np.ravel(y_train)

# %% [markdown]
# ## Hyperparameter Tuning 
# Exhaustive search over specified parameter values for the model using **GridSearch**

# %%
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error

parameters = {'n_neighbors':[5,7,9,11,13,15,17]}
knn = KNeighborsRegressor()
model_grid_search = GridSearchCV(knn, parameters,cv=10,scoring='neg_mean_absolute_error')
model_grid_search.fit(x_train, y_train)

# %%
# Print the best parameters and best score
print("Best Parameters: ", model_grid_search.best_params_)
#print("Best Mean Absolute Error: ", model_grid_search.best_score_)

# Get the best model
#best_model = model_grid_search.best_estimator_

# Evaluate the best model on the test set using Mean Squared Error
#y_pred = best_model.predict(x_test)
#mae = mean_absolute_error(y_test, y_pred)
#print("Mean Absolute Error with Best Model in Test : {:.2f}".format(mae))

# %% [markdown]
# ## Conclusions
#
# Finally, after the process of finding the best set of parameters for the KnnRegressor, it has been found that the better set of paramter is `{'n_neighbors': 15}`

# %% [markdown]
# # Traning and Evaluation of the Algorithm with the best set of Parameters

# %%
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error


# Creating the scikit-learn pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),               # Standardize features
    ('model', KNeighborsRegressor(n_neighbors=11)) #Knn regressor model 
])



# %%
# Seting up cross-validation on the training set
#cv = KFold(n_splits=10, shuffle=False, random_state=None)

# Defining the scorer for cross-validation
#scorer = make_scorer(mean_absolute_error, greater_is_better=False)

# Performing cross-validation on the training set
#cv_scores = cross_val_score(pipe, x_train, y_train, scoring=scorer, cv=cv)

# Display the average score and its standard deviation on the training set
#print(f'Cross-Validation Mean Absolute Error: {-cv_scores.mean():.4f} (± {cv_scores.std():.4f})')

# Train the pipeline on the entire training set
pipe.fit(x_train, y_trained)

# Make predictions on the test set
test_predictions = pipe.predict(x_test)

# Calculate and print the mean absolute error on the test set
test_mae = mean_absolute_error(y_test, test_predictions)
print(f'Test Mean Absolute Error: {test_mae:.4f}')

# %%
plot_actual_vs_predicted(y_test.values,test_predictions)

# %% [markdown]
# ## Conclusions
#
# - The model seems to be trying to minimize the error by reducing the prediction range to intermediate values.
# - This may reduce the error but do not seems to be very accurate. (Here we can see the drawbacks of Batch learning)

# %% [markdown]
# # Saving test results for comparison 

# %%
with open('batch_results.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['MAE KnnRegressor', test_mae])
