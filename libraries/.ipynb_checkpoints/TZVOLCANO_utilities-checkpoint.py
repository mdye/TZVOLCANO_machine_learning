# Function definitions are included in this file to improve code readibility 
# 
# Many of these functions were modified fron the excellent book
# Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd Edition
# by Aurélien Géron

import os

import pandas as pd
import numpy as np

# Plotting
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


# TensorFlow ≥2.0 is required
import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"


# Data pipeline, scaling, normalizing, etc
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer



def calculate_scalar(pandas_object, vector_fields_list, scale_minimum, scale_maximum):
    # create the string to evaluate
    strings =[]
    for vector_field in vector_fields_list:
        string_partial = "pandas_object['" + vector_field + "']**2"
        strings.append(string_partial)
    
    s = " + "
    string = s.join(strings)
    
    scalar_value = np.sqrt(eval(string))
    
    normalized_scalar_value = scale_np_data(scalar_value.to_numpy(), scale_minimum, scale_maximum)

    return normalized_scalar_value



def plot_series(series, n_steps, y=None, y_pred=None, x_label="$t$", y_label="$x(t)$"):
    plt.plot(series, ".-")
    if y is not None:
        plt.plot(n_steps, y, "bx", markersize=10)
    if y_pred is not None:
        plt.plot(n_steps, y_pred, "ro")
    plt.grid(True)
    if x_label:
        plt.xlabel(x_label, fontsize=16)
    if y_label:
        plt.ylabel(y_label, fontsize=16, rotation=0)
    plt.hlines(0, 0, 100, linewidth=1)
    plt.axis([0, n_steps + 1, -1, 1])
    
    
    
def plot_multiple_forecasts(X, Y, Y_pred, n_steps):
    n_steps = X.shape[1]
    ahead = Y.shape[1]
    plot_series(X[0, :, 0], n_steps)
    plt.plot(np.arange(n_steps, n_steps + ahead), Y[0, :, 0], "ro-", label="Actual")
    plt.plot(np.arange(n_steps, n_steps + ahead), Y_pred[0, :, 0], "bx-", label="Forecast", markersize=10)
    plt.axis([0, n_steps + ahead, -1, 1])
    plt.legend(fontsize=14)    
    
    
def plot_clusters(X, y=None):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=1)
    plt.xlabel("$x_1$", fontsize=14)
    plt.ylabel("$x_2$", fontsize=14, rotation=0)    

    
    
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    # Where to save the figures
    PROJECT_ROOT_DIR = "."
    IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images")
    os.makedirs(IMAGES_PATH, exist_ok=True)
    
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)    
    

    
def last_time_step_mse(Y_true, Y_pred):
    return keras.metrics.mean_squared_error(Y_true[:, -1], Y_pred[:, -1])


def scale_np_data(np_array, scale_minimum, scale_maximum):
    scaled_data_np = np.interp(np_array, (np_array.min(), np_array.max()), (scale_minimum, scale_maximum))

    return scaled_data_np


def get_RNN_training_sets(pandas_object, key, n_steps):
    np_data = pandas_object[key].to_numpy()

#     height_forecast = np_data[n_steps_training-n_steps_forecast:n_steps_training + n_steps_ahead]

    
    np_data = np_data[:n_steps+1]

    np_series = np.array([np_data])
    np_series = np_series[..., np.newaxis].astype(np.float32)

    np_training_data = np.array(np_series[:,:-1])
    np_labels = np.array(np_series[:,-1])    
    
    return np_training_data, np_labels


def get_RNN_forecast_sets(pandas_object, key, n_steps_training, n_steps_forecast, n_steps_ahead):
    
    np_data = pandas_object[key].to_numpy()

    # Get the tailing records in the data set 
#     np_data = np_data[n_steps_training - n_steps_forecast: n_steps_training + n_steps_ahead]
    np_data = np_data[-(n_steps_forecast + n_steps_ahead):]

    forecast_series = np.array([np_data])

    # Cast the data so they are float32
    forecast_series = forecast_series[..., np.newaxis].astype(np.float32)
    
    np_training_data = forecast_series[:, :n_steps_forecast]
    np_labels = forecast_series[:, n_steps_forecast:]
    
    return np_training_data, np_labels





def plot_centroids(centroids, weights=None, circle_color='w', cross_color='k'):
    print(centroids.size)
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='o', s=35, linewidths=8,
                color=circle_color, zorder=10, alpha=0.9)
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=2, linewidths=12,
                color=cross_color, zorder=11, alpha=1)
    


    
    
def plot_gaussian_mixture(clusterer, X, resolution=1000, show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    Z = -clusterer.score_samples(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z,
                 norm=LogNorm(vmin=1.0, vmax=30.0),
                 levels=np.logspace(0, 6, 96))
    plt.contour(xx, yy, Z,
                norm=LogNorm(vmin=1.0, vmax=30.0),
                levels=np.logspace(0, 6, 96),
                linewidths=1, colors='k')

    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z,
                linewidths=2, colors='r', linestyles='dashed')
    
    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)
    plot_centroids(clusterer.means_, clusterer.weights_)

    plt.xlabel("$x_1$", fontsize=14)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)    
        
        

        
# For K-means plotting
def plot_data(X):
    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)


def plot_decision_boundaries(clusterer, X, resolution=1000, show_centroids=True,
                             show_xlabels=True, show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                cmap="Pastel2")
    plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                linewidths=1, colors='k')
    plot_data(X)
    if show_centroids:
        plot_centroids(clusterer.cluster_centers_)

    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)        
        
        
def transform_data_for_kmeans(pandas_object, field_name):
    
    # Create a new pandas object to temporarily store the data before imputing
    data = pd.DataFrame()

    # Convert the Time variable to Seconds Since Epoch
    data["Seconds Since Epoch"] = pandas_object['Seconds Since Epoch']
    data[field_name] = pandas_object[field_name]
    
    
    # Define a pipline to clean numerical data
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="most_frequent")),
        ('std_scaler', StandardScaler()),
    ])

    # Run the pipeline
    data_imputed = num_pipeline.fit_transform(data)    
    
    return data_imputed        




def get_rnn_model(n_steps_ahead):
    np.random.seed(42)
    tf.random.set_seed(42)

#     model = keras.models.Sequential([
#         keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
#         keras.layers.BatchNormalization(),
#         keras.layers.SimpleRNN(20, return_sequences=True),
#         keras.layers.BatchNormalization(),
#         keras.layers.TimeDistributed(keras.layers.Dense(n_steps_ahead))
#     ])    
    
#     model.compile(loss="mse", optimizer="adam", metrics=[last_time_step_mse])





# This is the model that produces decent results
#     model = keras.models.Sequential([
#         keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
#         keras.layers.SimpleRNN(20, return_sequences=True),
#         keras.layers.TimeDistributed(keras.layers.Dense(n_steps_ahead))
#     ])

#     model.compile(loss="mse", optimizer=keras.optimizers.Adam(learning_rate=0.01), metrics=[last_time_step_mse])
    

    

#     model = keras.models.Sequential([
#         keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
#         keras.layers.SimpleRNN(20),
#         keras.layers.Dense(n_steps_ahead)
#     ])
    
#     model.compile(loss="mse", optimizer="adam")


#     model = keras.models.Sequential([
#         keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
#         keras.layers.SimpleRNN(20),
#         keras.layers.Dense(n_steps_ahead)
#     ])
    
#     model.compile(loss="mse", optimizer=keras.optimizers.Adam(learning_rate=0.01), metrics=[last_time_step_mse])



    
#     model = keras.models.Sequential([
#         keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
#         keras.layers.Dropout(rate=0.2),
#         keras.layers.SimpleRNN(20, return_sequences=True),
#         keras.layers.TimeDistributed(keras.layers.Dense(n_steps_ahead))
#     ])

#     model.compile(loss="mse", optimizer=keras.optimizers.Adam(learning_rate=0.01), metrics=[last_time_step_mse])
    
    
#     model = keras.models.Sequential([
#         keras.layers.LSTM(20, return_sequences=True, input_shape=[None, 1]),
#         keras.layers.LSTM(20, return_sequences=True),
#         keras.layers.TimeDistributed(keras.layers.Dense(n_steps_ahead))
#     ])

#     model.compile(loss="mse", optimizer=keras.optimizers.Adam(learning_rate=0.01), metrics=[last_time_step_mse])

    
#     model = keras.models.Sequential([
#         keras.layers.LSTM(20, return_sequences=True, input_shape=[None, 1]),
#         keras.layers.LSTM(20, return_sequences=True),
#         keras.layers.TimeDistributed(keras.layers.Dense(n_steps_ahead))
#     ])

#     model.compile(loss="mse", optimizer="adam", metrics=[last_time_step_mse])

    
    model = keras.models.Sequential([
        keras.layers.Conv1D(filters=20, kernel_size=4, strides=2, padding="valid",
                            input_shape=[None, 1]),
        keras.layers.GRU(20, return_sequences=True),
        keras.layers.Dropout(rate=0.2),
        keras.layers.GRU(20, return_sequences=True),
        keras.layers.TimeDistributed(keras.layers.Dense(n_steps_ahead))
    ])

#     model.compile(loss="mse", optimizer="adam", metrics=[last_time_step_mse])    
    model.compile(loss="mse", optimizer=keras.optimizers.Adam(learning_rate=0.01), metrics=[last_time_step_mse])
    
    return model


