import datetime as dt
import pytz
import multiprocessing as mp

import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


def train(n_neurons, look_back, n_epochs, model_path):

    df = pd.read_csv("splitted_interpolated/1-Aranmore.csv")

    df['time'] = pd.to_datetime(df['time'])

    df['timeinseconds'] = df['time'].map(pd.Timestamp.timestamp)

    # utc, because if not it would be shifted 1 hour after converting
    df['time_back'] = df['timeinseconds'].map(lambda x: dt.datetime.fromtimestamp(x, tz=pytz.utc).replace(tzinfo=None))

    df = df[['timeinseconds', "Water_Level"]]

    time = df['timeinseconds'].values
    waterlevel = df['Water_Level'].values

    data_length = len(df)
    X_train = time[0:int(data_length*0.9)]
    y_train = waterlevel[0:int(data_length*0.9)]

    X_test = time[int(data_length*0.9):]
    y_test = waterlevel[int(data_length*0.9):]


    n_features = 1

    train_series = y_train.reshape((len(y_train), n_features))
    test_series  = y_test.reshape((len(y_test), n_features))

    """fig, ax = plt.subplots(1, 1, figsize=(15, 4))
    ax.plot(X_train,y_train, lw=3, label='train data')
    ax.plot(X_test, y_test,  lw=3, label='test data')
    ax.legend(loc="lower left")
    plt.show()"""



    train_generator = TimeseriesGenerator(train_series, train_series,
                                          length        = look_back,
                                          sampling_rate = 1,
                                          stride        = 1,
                                          batch_size    = 10)

    test_generator = TimeseriesGenerator(test_series, test_series,
                                          length        = look_back,
                                          sampling_rate = 1,
                                          stride        = 1,
                                          batch_size    = 10)

    model = Sequential()
    model.add(LSTM(n_neurons, input_shape=(look_back, n_features)))
    model.add(Dense(1))

    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                    optimizer=tf.keras.optimizers.Adam(),
                    metrics=[tf.keras.metrics.MeanAbsoluteError()])
    print(f"training model {model_path}...")
    history = model.fit(train_generator, epochs=n_epochs,
                            verbose=0)

    model.save(model_path)
    print(f"Saved model {model_path}.")


if __name__ == '__main__':

    p1_neurons = 8
    p1_lookback = 144
    p1_epochs = 10
    p1 = mp.Process(target=train, args=(p1_neurons, p1_lookback, p1_epochs, f"hdf5/LSTM-{p1_neurons}-Dense1-aranmore-{p1_epochs}epochs-{p1_lookback}lookback.h5"))

    p2_neurons = 2
    p2_lookback = 144
    p2_epochs = 10
    p2 = mp.Process(target=train, args=(p2_neurons, p2_lookback, p2_epochs, f"hdf5/LSTM-{p2_neurons}-Dense1-aranmore-{p2_epochs}epochs-{p2_lookback}lookback.h5"))

    p3_neurons = 8
    p3_lookback = 144
    p3_epochs = 20
    p3 = mp.Process(target=train, args=(p3_neurons, p3_lookback, p3_epochs,
                                        f"hdf5/LSTM-{p3_neurons}-Dense1-aranmore-{p3_epochs}epochs-{p3_lookback}lookback.h5"))

    p4_neurons = 8
    p4_lookback = 144
    p4_epochs = 10
    p4 = mp.Process(target=train, args=(p4_neurons, p4_lookback, p4_epochs,
                                        f"hdf5/LSTM-{p4_neurons}-Dense1-aranmore-{p4_epochs}epochs-{p4_lookback}lookback.h5"))

    p1.start()
    p2.start()
    p3.start()
    p4.start()