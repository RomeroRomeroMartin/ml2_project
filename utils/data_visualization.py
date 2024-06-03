import matplotlib.pyplot as plt
from statistics import mean
from rich import print
import numpy as np

import ipywidgets as widgets
from IPython.display import display

from matplotlib import gridspec

def plot_concept_drift(stream, drifts=None):
    fig, ax1 = plt.subplots(figsize=(20, 5), tight_layout=True)
    ax1.grid()
    ax1.plot(stream, label='Stream', color='seagreen', alpha=0.6)
    ax1.yaxis.grid(False)
    
    if drifts is not None:
        for drift_detected in drifts:
            ax1.axvline(drift_detected, color='rosybrown')
    
    plt.show()

def plot_metric_history(history, metric='MAE'):
    plt.figure(figsize=(10, 6))
    plt.plot(history['metric_values'], label=metric, color='seagreen')
    plt.xlabel('Iteration')

    plt.ylabel(metric)
    plt.title('Progression of ' + metric +' over iterations')
    plt.legend()
    plt.show()

def plot_interactive_hist(data, bins=30, title='% Silica Concentrate distribution', 
                          x_lab='Silica Concentrate %', y_lab='Count', figsize=(10,5), x_lim = None):
    def plot_histogram(n):
    
        plt.figure(figsize=figsize)
        plt.hist(data[:n], bins=30, color='seagreen', edgecolor='black', alpha=0.7)
        plt.title(title)
        plt.xlabel(x_lab)
        plt.ylabel(y_lab)
        plt.grid(axis='y', alpha=0.75)
        if x_lim:
            plt.xlim(*x_lim)
        plt.show()    

    widgets.interact(plot_histogram, n=widgets.IntSlider(min=1, max=len(data), step=1000, value=1000))

def plot_histogram(data, title='Predictions distribution' , x_lab='Silica Concentrate % prediction' , 
                   y_lab='Count', figsize=(10,6), x_lim = None):
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=30, color='seagreen', edgecolor='black', alpha=0.7)
    plt.title(title)
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.grid(axis='y', alpha=0.75)
    if x_lim:
        plt.xlim(*x_lim)
    plt.show()

def plot_actual_vs_predicted(real_y, predicted_y, figsize=(20, 6), y_lim=(0,6)):
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    
    # Plot actual vs predicted values
    axs[0].scatter(real_y, predicted_y, color='seagreen', alpha=0.7)
    axs[0].plot([min(real_y), max(real_y)], [min(real_y), max(real_y)], linestyle='--', color='black', linewidth=2)
    axs[0].set_xlabel('Actual Values')
    axs[0].set_ylabel('Predicted Values')
    axs[0].set_title('Actual vs Predicted Values')
    axs[0].set_ylim(*y_lim)
    
    # Plot actual vs predicted values against instance number
    instances = np.arange(len(real_y))
    axs[1].plot(instances, real_y, label='Actual Values', marker='o', color='lightblue')
    axs[1].plot(instances, predicted_y, label='Predicted Values', marker='o', color='seagreen', alpha=0.7)
    axs[1].set_xlabel('Instance Number')
    axs[1].set_ylabel('Values')
    axs[1].set_title('Actual vs Predicted Values')
    axs[1].legend()
    axs[1].set_ylim(*y_lim)

    plt.tight_layout()
    plt.show()

def plot_means_and_stds_with_labels(data_dict, figsize=(10,7)):
    """
    This function takes a dictionary where each key is a string (label) and its value is a list of numbers. 
    It calculates the mean and standard deviation for the list of numbers associated with each key and 
    plots these statistics in a bar chart, using the keys as labels for the bars.
    
    Parameters:
    - data_dict: A dictionary with string keys and list of numbers as values.
    """
    labels = list(data_dict.keys())
    means = [np.mean(data_dict[label]) for label in labels]
    stds = [np.std(data_dict[label]) for label in labels]
    indices = np.arange(len(labels))
    
    # Plotting
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(indices, means, yerr=stds, capsize=5, alpha=0.5, color='seagreen')
    
    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Mean Values')
    ax.set_title('Mean and Standard Deviation for each Algorithm')
    ax.set_xticks(indices)
    ax.set_xticklabels(labels)
    ax.yaxis.grid(True)  # Add horizontal grid lines for better readability
    
    # Display the plot
    plt.tight_layout()
    plt.show()


def plot_many_metrics(metrics, colors=None, figsize=(10,7), y_lim=None):
    """
    Plot the progression of multiple metrics over iterations.

    Parameters:
    - histories: A dictionary where keys are metric names and values are lists of metric values.
    - colors: A list of colors for each metric line. If not provided, a default set will be used.
    """
    plt.figure(figsize=figsize)

    if colors is None:
        # Use a default color cycle if no colors are provided
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i, (metric_name, metric_values) in enumerate(metrics.items()):
        plt.plot(metric_values, label=metric_name, color=colors[i % len(colors)], alpha=0.7)

    plt.xlabel('Iteration')
    plt.ylabel('Metric Value')
    plt.title('Progression of Metrics over Iterations')
    plt.legend()
    if y_lim:
        plt.ylim(*y_lim)
    plt.show()

def plot_online_batch_comparison(data, metric='MAE', algorithm='', constant_line=None, figsize=(20,10), y_lim=None):
    plt.figure(figsize=figsize)
    plt.plot(data, label=metric, color='seagreen', alpha=0.7)
    
    if constant_line:
        plt.axhline(y=constant_line, color='rosybrown', linestyle='--', label='Batch Baseline')
    
    plt.xlabel('Iteration')
    plt.ylabel(metric)
    plt.title('Online vs Batch ' + algorithm+ ' ' + metric + ' comparison')
    plt.legend()
    if y_lim:
        plt.ylim(*y_lim)
    plt.show()

def plot_time_history(data, algorithm='*Algorithm*'):
    """
    Plot the time history for predictions of an algorithm and display the total time.
    
    Parameters:
    - data: A list of durations (in seconds) it took for each prediction.
    - algorithm: A string representing the name of the algorithm.
    """
    total_time = sum(data)  # Sum the list to get the total time
    
    plt.figure(figsize=(10, 6))
    plt.plot(data, label='Seconds', color='seagreen', alpha=0.7)
    plt.xlabel('Iteration')
    plt.ylabel('Seconds')
    plt.suptitle(f'Time to produce a prediction for {algorithm} (Seconds)')
    plt.title(f'Total time: {total_time:.2f} seconds')
    plt.legend()
    plt.show()