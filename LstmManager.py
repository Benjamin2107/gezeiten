import os

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf


class TidePrediction:
    def __init__(self, model_number, timestamp, model_path="single/", data_path="splitted_interpolated/"):
        self.model_path = model_path
        self.data_path = data_path
        self.model_number = model_number
        self.timestamp = timestamp
        self.model = self.load_model()
        self.lookback_data = self.load_data()

    def load_model(self):
        """
        Loads the hdf5 file from a model
        :param model_number: Integer from 0 to 32
        :return: model
        """
        model_files = os.listdir(self.model_path)
        model_name = [f for f in model_files if f.startswith(str(self.model_number))][0]
        return tf.keras.models.load_model(f'{self.model_path}{model_name}')

    def load_data(self):
        fname = [f for f in os.listdir(self.data_path) if f.startswith(str(self.model_number) + "-")][0]
        df = pd.read_csv(f"{self.data_path}{fname}")
        df = df.tail(self.model.input_shape[1])

        return df[["time", "Water_Level"]]

    def generate_time_series(self):
        self.timestamp = pd.Timestamp(self.timestamp).floor("10 min")
        reference_timestamp = pd.Timestamp(self.lookback_data["time"].iloc[-1]).floor("10 min")
        timestamps = pd.date_range(start=reference_timestamp, end=self.timestamp, freq="10min")

        return timestamps


class TidePredictionSingleOutput(TidePrediction):

    def predict_series(self):
        timestamps = self.generate_time_series()

        extrapolation = list()
        seed_batch = self.lookback_data["Water_Level"].values.reshape((1, self.lookback_data["Water_Level"].size, 1))
        current_batch = seed_batch
        for i in range(timestamps.size):
            predicted_val = self.model.predict(current_batch, verbose=0)[0]
            extrapolation.append(predicted_val)
            current_batch = np.append(current_batch[:, 1:, :], [[predicted_val]], axis=1)

        return extrapolation

    def predict_timestamp(self):
        all_predicted_vals = self.predict_series()
        predicted_val = all_predicted_vals[-1][0]
        print(f"The estimated Water Level for Station {self.model_number} is {predicted_val} m.")
        return predicted_val


class TidePredictionMultiOutput(TidePrediction):
    def __init__(self, model_number, timestamp, minmaxscaler_path="minmaxscaler_720_lookback/"):
        super().__init__(model_number, timestamp, model_path="models_720_lookback/")
        self.minmaxscaler_path = minmaxscaler_path
        self.timestamps = self.generate_time_series()
        self.minmaxscaler = self.load_minmaxscaler()

    def load_minmaxscaler(self):
        return joblib.load(self.minmaxscaler_path + f"{self.model_number}.save")

    def predict_series(self):
        extrapolation = list()
        seed_batch = self.lookback_data["Water_Level"].values.reshape((1, self.lookback_data["Water_Level"].size, 1))
        current_batch = self.transform_values(seed_batch)
        for i in range(self.timestamps.size):
            predicted_val = self.model.predict(current_batch, verbose=0)[0]
            extrapolation.append(predicted_val)
            current_batch = np.append(current_batch[:, self.model.output_shape[1]:, :],
                                      predicted_val.reshape(1, self.model.output_shape[1], 1 ),
                                      axis=1)

        return extrapolation

    def predict_timestamp(self):
        all_predicted_vals = self.predict_series()
        predicted_val = all_predicted_vals[self.timestamps.size - 1][0]
        inverse_value = self.inverse_transform_value(predicted_val)
        print(f"The estimated Water Level for Station {self.model_number} is {inverse_value.astype(np.float16)} m.")
        return inverse_value

    def transform_values(self, values):
        values = values.reshape(-1, 1)
        values = self.minmaxscaler.transform(values)
        values = values.reshape((1, self.lookback_data["Water_Level"].size, 1))
        return values

    def inverse_transform_value(self, value):
        value = self.minmaxscaler.inverse_transform(value.reshape(-1, 1))
        return value[0][0]
