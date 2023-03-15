import multiprocessing as mp
import os

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

from WindowGenerator import generate_window, multi_step_plot, plot_train_history


def train(station_ids, neurons_first_layer, neurons_second_layer, look_back, look_front, epochs, evaluation_interval=200, ):

    for station_id in station_ids:
        batch_size = 256
        buffer_size = 10000
        train_val_split = 140000
        train_test_split = 150000
        patience = 5

        seed = 13
        tf.random.set_seed(seed)

        model_path = f"multioutput_models/{station_id}-{look_back}-LSTM{neurons_first_layer}-LSTM{neurons_second_layer}relu-Dense{look_front}-{epochs}epochs.h5"

        station_files = os.listdir("splitted_interpolated/")
        file = [f for f in station_files if f.startswith(str(station_id))][0]
        df = pd.read_csv("splitted_interpolated/" + file)
        df['time'] = pd.to_datetime(df['time'])

        df['timeinseconds'] = df['time'].map(pd.Timestamp.timestamp)

        df = df[['timeinseconds', "Water_Level"]]

        dataset = df['Water_Level']
        dataset.index = df["timeinseconds"]

        dataset = dataset.values
        train_data = dataset[:train_val_split].reshape(-1, 1)
        test_data = dataset[train_val_split:].reshape(-1, 1)
        minmaxscaler = MinMaxScaler()
        train_data = minmaxscaler.fit_transform(train_data)
        joblib.dump(minmaxscaler, f"minmaxscaler/{station_id}.save")
        minmaxscaler = joblib.load(f"minmaxscaler/{station_id}.save")
        test_data = minmaxscaler.transform(test_data)
        dataset = np.concatenate((train_data, test_data))

        x_train_multi, y_train_multi = generate_window(dataset, dataset, 0,
                                                         train_val_split, look_back,
                                                         look_front)
        x_val_multi, y_val_multi = generate_window(dataset, dataset,
                                                     train_val_split, train_test_split, look_back,
                                                     look_front)

        x_test_multi, y_test_multi = generate_window(dataset, dataset,
                                                   train_test_split, None, look_back,
                                                   look_front)

        """print(x_train_multi.shape,
              y_train_multi.shape,
              'Single window of past history : {}'.format(x_train_multi[0].shape),
              'Target Water Level to predict : {}'.format(y_train_multi[0].shape),
              sep='\n')"""

        train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
        train_data_multi = train_data_multi.cache().shuffle(buffer_size).batch(batch_size).repeat()

        val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
        val_data_multi = val_data_multi.batch(batch_size).repeat()

        """for x, y in train_data_multi.take(1):
            multi_step_plot(x[0], y[0], np.array([0]))"""

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.LSTM(neurons_first_layer,
                                       return_sequences=True,
                                       input_shape=(look_back, 1)))
        model.add(tf.keras.layers.LSTM(neurons_second_layer))
        model.add(tf.keras.layers.Dense(look_front))

        model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mae')

        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        print(f"Training on {model_path}...")
        history = model.fit(train_data_multi,
                            epochs=epochs,
                            steps_per_epoch=evaluation_interval,
                            validation_data=val_data_multi,
                            validation_steps=evaluation_interval,
                            callbacks=[early_stopping],
                            verbose=0)

        test_data_multi = tf.data.Dataset.from_tensor_slices((x_test_multi, y_test_multi))
        test_data_multi = test_data_multi.batch(batch_size).repeat()

        # Evaluate the model on the test data
        test_loss = model.evaluate(test_data_multi, steps=len(x_test_multi) // batch_size, verbose=0)
        print(f"Test loss for model {model_path} : {test_loss}")

        model.save(model_path)
        print(f"Saved Model: {model_path}")
        #plot_train_history(history, model_path)

        """model = tf.keras.models.load_model(model_path)
        for x, y in val_data_multi.take(3):
            multi_step_plot(x[0], y[0], model.predict(x)[0])"""


if __name__ == "__main__":
    p1_l1_neurons = 128
    p1_l2_neurons = 32
    p1_lookback = 720
    p1_lookfront = 144
    p1_epochs = 50
    p1_station_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    p1 = mp.Process(target=train, args=(p1_station_ids, p1_l1_neurons, p1_l2_neurons,
                                        p1_lookback, p1_lookfront, p1_epochs,))

    p2_l1_neurons = 128
    p2_l2_neurons = 32
    p2_lookback = 720
    p2_lookfront = 144
    p2_epochs = 50
    p2_station_ids = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    p2 = mp.Process(target=train, args=(p2_station_ids, p2_l1_neurons, p2_l2_neurons,
                                        p2_lookback, p2_lookfront, p2_epochs,))

    p3_l1_neurons = 128
    p3_l2_neurons = 32
    p3_lookback = 720
    p3_lookfront = 144
    p3_epochs = 50
    p3_station_ids = [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
    p3 = mp.Process(target=train, args=(p3_station_ids, p3_l1_neurons, p3_l2_neurons,
                                        p3_lookback, p3_lookfront, p3_epochs,))

    p1.start()
    p2.start()
    p3.start()

    """train(p2_station_ids, p2_l1_neurons, p2_l2_neurons,
                                        p2_lookback, p2_lookfront, p2_epochs,)"""