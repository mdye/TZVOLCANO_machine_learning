import numpy as np

# TensorFlow ≥2.0 is required
import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"



def last_time_step_mse(Y_true, Y_pred):
    return keras.metrics.mean_squared_error(Y_true[:, -1], Y_pred[:, -1])




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
