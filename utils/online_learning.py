from river import compose
from river import preprocessing
from river import tree
from river import metrics
from river import feature_extraction as fx
from river import stats
from river import neighbors
from river import stream
from statistics import mean
import time


from rich import print


def build_pipeline(model, features_to_mean, to_discard):
    # This function helps on building the needed pipeline for
    # the mining problem

    # Add aggregators to the pipeline:
    for i, feature in enumerate(features_to_mean):
        if i == 0:
            pipeline = fx.Agg(on=feature, by='date', how= stats.Mean())
        else:
            pipeline += fx.Agg(on=feature, by='date', how= stats.Mean())


    pipeline |= compose.Discard(*to_discard)
    pipeline |= preprocessing.StandardScaler()
    pipeline |= model

    return pipeline

def update_agg_components(pipeline, x):
    x_transformed = {}
    # This function iterates through the transformers at the pipeline's transformerUnion
    # and updates the aggregators with the values of x
    for agg in list(pipeline.steps.values())[0]:
        agg.learn_one(x)
        x_transformed.update(agg.transform_one(x))
    return x_transformed

def update_standard_scaler(pipeline, x_transformed):
    # Updates directly the StandardScaler inside the pipeline
    standard_scaler = pipeline['StandardScaler']
    standard_scaler.learn_one(x_transformed)

def update_pipeline_state(pipeline, x):
    # Wrapper function to update the aggregators and scaler of the given pipeline
    x_transformed = update_agg_components(pipeline,x)
    update_standard_scaler(pipeline, x_transformed)

def train_online(pipeline, data_stream, metric=metrics.MAE(), show_metric=True, show_predictions=False):
    history = {'metric_values':[], 'predictions':[], 'times':[]}
    
    old_date = None
    hour_pred_batch = []
    # Variables to keep (x,y) values of last iteration
    last_y = 0
    last_x = 0

    for i, (x, y) in enumerate(data_stream):
        # Let's compute how much time it takes to produce a prediction
        # since a new instance is available
        start_time = time.time()
        
        current_date = x['date']
        old_date = current_date if old_date is None else old_date
    
        if current_date == old_date:
            last_y = y
            last_x = x

            # Make a prediction
            # To use the means agg by date, including the current feature
            # the aggregators must be updated with the instance beforehand:
            
            update_pipeline_state(pipeline, x) # Update the Means agg by date with x values. Also update the scaler (No model learning here)
            y_pred = pipeline.predict_one(x)
            end_time = time.time()
            hour_pred_batch.append(y_pred)
    
        else: # New hour batch
            for pred in hour_pred_batch: # Update metric with all predictions
                metric.update(last_y, pred)
        
            # Learn is only done when the hour changes and the process is finished (We now have the real y (last_y))
            pipeline.learn_one(last_x, last_y) #Learn the last batch before changing
            hour_pred_batch = [] # Empty the pred batch
            old_date = current_date # Set the new datetime to detect future changes

            # Process the current instance
            update_pipeline_state(pipeline, x)
            y_pred = pipeline.predict_one(x)
            end_time = time.time()
            hour_pred_batch.append(y_pred)
    
            # Print the just updated metric
            if show_metric:
                print(f"Iteration {i + 1}, MAE: {metric.get()}")
            #Save metrics and 
            history['metric_values'].append(metric.get())
        
        # Collect the computing time and all predictions to plot the distribution
        history['times'].append(end_time - start_time)
        history['predictions'].append(y_pred)

        
        if show_predictions:
            print(f"Prediction at {current_date} is {y_pred}% of Silica concentrate")

     # Handle the last batch after the loop
    for pred in hour_pred_batch:  # Update metric with all predictions of the last batch
        metric.update(last_y, pred)
    
    pipeline.learn_one(last_x, last_y)
    
    # Ensure the metric is updated for the last batch
    print(f"Final Iteration, MAE: {metric.get()}")
    history['metric_values'].append(metric.get())
    
    return history
