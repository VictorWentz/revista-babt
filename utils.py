""" Paper Title: Analysis of Artificial Neural Networks Models for
    Forecasting Solar Photovoltaic Generation
    Authors: Wentz VH; Maciel JN; Ledesma JJG; Ando Junior, OH
    Objetive "train.py": Create and train ANN models
    Updated: 08/03/2021
"""


import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import math

EPSILON = 1e-10


def _error(actual: np.ndarray, predicted: np.ndarray):
    """ Simple error """
    return actual - predicted


def _percentage_error(actual: np.ndarray, predicted: np.ndarray):
    """
    Percentage error
    Note: result is NOT multiplied by 100
    """
    return _error(actual, predicted) / (actual + EPSILON)


def mse(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Squared Error """
    return np.mean(np.square(_error(actual, predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray):
    """ Root Mean Squared Error """
    return np.sqrt(mse(actual, predicted))


def nrmse(actual: np.ndarray, predicted: np.ndarray):
    """ Normalized Root Mean Squared Error """
    return rmse(actual, predicted) / (actual.max() - actual.min())


def mape2(actual: np.ndarray, predicted: np.ndarray):
    """
    Mean Absolute Percentage Error
    Properties:
        + Easy to interpret
        + Scale independent
        - Biased, not symmetric
        - Undefined when actual[t] == 0
    Note: result is NOT multiplied by 100
    """
    return np.mean(np.abs(_percentage_error(actual, predicted)))

# With GHI


def split_sequences_ghi(sequences, n_steps):

    X, y = list(), list()
    try:
        for i in range(len(sequences)):
            # find the end of this pattern
            end_ix = i + n_steps
            # check if we are beyond the dataset
            if end_ix > len(sequences):
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, -1]
            X.append(seq_x)
            y.append(seq_y)

    except IndexError:
        return np.array(X), np.array(y)

    return np.array(X), np.array(y)


# Without GHI
def split_sequences(sequences, n_steps):
    X, y = list(), list()

    try:
        for i in range(len(sequences)):
            # find the end of this pattern
            end_ix = i + n_steps
            # check if we are beyond the dataset
            if end_ix > len(sequences):
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
            X.append(seq_x)
            y.append(seq_y)
    except IndexError:
        return np.array(X), np.array(y)
    return np.array(X), np.array(y)


def get_data(weatherPath: str,
             irradiancePath: str,
             timeStamp: int,
             drops: list = None) -> pd.DataFrame:
    """Load pedro's data

    Args:
        weatherPath (str): Path to Folsom_weather.csv
        irradiancePath (str): Path to Folsom_irradiance.csv
        timeStamp (int): step of each measure
        drops (list, optional): Columns to drop. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame with pedro's data
    """
    if timeStamp == 60:
        if drops is None:

            dataWeather = pd.read_csv(
                weatherPath, index_col=0, parse_dates=True)
            dataWeather["month"] = dataWeather.index.month
            dataWeather["day"] = dataWeather.index.day
            dataWeather["hour"] = dataWeather.index.hour

            dataIrradiance = pd.read_csv(irradiancePath, index_col=0, usecols=[0, 1],
                                         parse_dates=True)

            data = pd.concat([dataWeather, dataIrradiance], axis=1)
            data.dropna(inplace=True)
            data = data[data.index.minute % timeStamp == 0]

            return data

        elif type(drops) is list:
            try:
                dataWeather = pd.read_csv(
                    weatherPath, index_col=0, parse_dates=True)
                dataWeather["month"] = dataWeather.index.month
                dataWeather["day"] = dataWeather.index.day
                dataWeather["hour"] = dataWeather.index.hour

                dataWeather.drop(drops, axis=1, inplace=True)

                dataIrradiance = pd.read_csv(irradiancePath, index_col=0, usecols=[0, 1],
                                             parse_dates=True)

                data = pd.concat([dataWeather, dataIrradiance], axis=1)
                data.dropna(inplace=True)
                data = data[data.index.minute % timeStamp == 0]

                return data

            except KeyError:
                print("[INFO] Select columns did not exist in data")
                print("[INFO] No columns were removed")

                dataWeather = pd.read_csv(
                    weatherPath, index_col=0, parse_dates=True)
                dataWeather["month"] = dataWeather.index.month
                dataWeather["day"] = dataWeather.index.day
                dataWeather["hour"] = dataWeather.index.hour

                dataIrradiance = pd.read_csv(irradiancePath, index_col=0, usecols=[0, 1],
                                             parse_dates=True)

                data = pd.concat([dataWeather, dataIrradiance], axis=1)
                data.dropna(inplace=True)
                data = data[data.index.minute % timeStamp == 0]

                return data

        else:
            raise TypeError("Drops needs to be list or None")
    # não 60
    else:

        if drops is None:

            dataWeather = pd.read_csv(
                weatherPath, index_col=0, parse_dates=True)
            dataWeather["month"] = dataWeather.index.month
            dataWeather["day"] = dataWeather.index.day
            dataWeather["hour"] = dataWeather.index.hour
            dataWeather["minute"] = dataWeather.index.minute

            dataIrradiance = pd.read_csv(irradiancePath, index_col=0, usecols=[0, 1],
                                         parse_dates=True)

            data = pd.concat([dataWeather, dataIrradiance], axis=1)
            data.dropna(inplace=True)
            data = data[data.index.minute % timeStamp == 0]

            return data

        elif type(drops) is list:
            try:
                dataWeather = pd.read_csv(
                    weatherPath, index_col=0, parse_dates=True)
                dataWeather["month"] = dataWeather.index.month
                dataWeather["day"] = dataWeather.index.day
                dataWeather["hour"] = dataWeather.index.hour
                dataWeather["minute"] = dataWeather.index.minute
                dataWeather.drop(drops, axis=1, inplace=True)

                dataIrradiance = pd.read_csv(irradiancePath, index_col=0, usecols=[0, 1],
                                             parse_dates=True)

                data = pd.concat([dataWeather, dataIrradiance], axis=1)
                data.dropna(inplace=True)
                data = data[data.index.minute % timeStamp == 0]

                return data

            except KeyError:
                print("[INFO] Select columns did not exist in data")
                print("[INFO] No columns were removed")

                dataWeather = pd.read_csv(
                    weatherPath, index_col=0, parse_dates=True)
                dataWeather["month"] = dataWeather.index.month
                dataWeather["day"] = dataWeather.index.day
                dataWeather["hour"] = dataWeather.index.hour
                dataWeather["minute"] = dataWeather.index.minute

                dataIrradiance = pd.read_csv(irradiancePath, index_col=0, usecols=[0, 1],
                                             parse_dates=True)

                data = pd.concat([dataWeather, dataIrradiance], axis=1)
                data.dropna(inplace=True)
                data = data[data.index.minute % timeStamp == 0]

                return data

        else:
            raise TypeError("Drops needs to be list or None")


def data_train_test(data, year, ghi):
    train = data[data.index.year != year]
    test = data[data.index.year == year]

    trainNorm = 0.1 + 0.8 * ((train - train.min()) /
                             (train.max() - train.min()))
    testNorm = 0.1 + 0.8 * ((test - test.min()) / (test.max() - test.min()))

    TrainDataset = np.hstack([trainNorm[i].values.reshape(
        (len(trainNorm[i]), 1)) for i in trainNorm.columns])
    TestDataset = np.hstack([testNorm[i].values.reshape(
        (len(testNorm[i]), 1)) for i in testNorm.columns])

    if ghi:
        # Train
        X_train, y_train = split_sequences_ghi(TrainDataset, 1)

        # Validation
        X_val, y_val = split_sequences_ghi(
            TestDataset[:int(len(TestDataset)/2)], 1)

        # test
        X_test, y_test = split_sequences_ghi(
            TestDataset[int(len(TestDataset)/2):], 1)

    else:
        # Train
        X_train, y_train = split_sequences(TrainDataset, 1)

        # Validation
        X_val, y_val = split_sequences(
            TestDataset[:int(len(TestDataset)/2)], 1)

        # test
        X_test, y_test = split_sequences(
            TestDataset[int(len(TestDataset)/2):], 1)

    n_in = X_train.shape[1] * X_train.shape[2]

    X_train = X_train.reshape((X_train.shape[0], n_in))

    X_val = X_val.reshape((X_val.shape[0], n_in))

    X_test = X_test.reshape((X_test.shape[0], n_in))

    return n_in, X_train, y_train, X_val, y_val, X_test, y_test


def print_result(model, X_test, y_test):
    y_pred = model.predict(X_test)

    print(f"RMSE: {rmse(y_test, y_pred.squeeze())}")
    print(f"nRMSE: {nrmse(y_test,y_pred.squeeze())}")
    print(f"MAPE: {mape2(y_test,y_pred.squeeze())} %")
    print(f"R2:{r2_score(y_test,y_pred.squeeze())}")


def data_for_graph_per_day(data_all: pd.DataFrame, year: int, month: str, days: list, ghi: bool):
    """

    :param data_all: Pandas dataframe with all data
    :param year: Integer, year to plot
    :param month: String, month you want plot ex: "5" - may
    :param days: List, List[0] - First day, List[1] - Last day
    :param ghi: Bool, True if you want ghi as feature
    :return: n_in, X_test, y_test
    """

    month_norm = {"1": 0.1, "2": 0.17272727, "3": 0.24545455, "4": 0.31818182, "5": 0.39090909,
                  "6": 0.46363636, "7": 0.53636364, "8": 0.60909091, "9": 0.68181818, "10": 0.75454545,
                  "11": 0.82727273, "12": 0.9}

    data = data_all[data_all.index.year == year]
    data = data.query(f"month == {int(month)} and {days[0]} < day < {days[1]}")

    dataNorm = 0.1 + 0.8 * ((data - data.min()) / (data.max() - data.min()))

    testDataSet = np.hstack([dataNorm[i].values.reshape(
        (len(dataNorm[i]), 1)) for i in dataNorm.columns])

    if ghi:
        # test
        X_test, y_test = split_sequences_ghi(testDataSet, 1)

    else:
        # test
        X_test, y_test = split_sequences(testDataSet, 1)

    n_in = X_test.shape[1] * X_test.shape[2]

    X_test = X_test.reshape((X_test.shape[0], n_in))

    X_test = np.nan_to_num(X_test, nan=month_norm[month])

    return n_in, X_test, y_test


def data_for_graph_per_interval(data_all: pd.DataFrame, year: int, month: str, interval: list, ghi: bool):
    """

    :param data_all: Pandas dataframe with all data
    :param year: Integer, year to plot
    :param month: String, month you want plot ex: "5" - may
    :param interval: List, list of interval plot
    :param ghi: Bool, True if you want ghi as feature
    :return: n_in, X_test, y_test
    """

    month_norm = {"1": 0.1, "2": 0.17272727, "3": 0.24545455, "4": 0.31818182, "5": 0.39090909,
                  "6": 0.46363636, "7": 0.53636364, "8": 0.60909091, "9": 0.68181818, "10": 0.75454545,
                  "11": 0.82727273, "12": 0.9}

    data = data_all[(data_all.index.year == year) & (
        data_all.index.month == int(month))][interval[0]:interval[1]]

    dataNorm = 0.1 + 0.8 * ((data - data.min()) / (data.max() - data.min()))

    testDataSet = np.hstack([dataNorm[i].values.reshape(
        (len(dataNorm[i]), 1)) for i in dataNorm.columns])

    if ghi:
        # test
        X_test, y_test = split_sequences_ghi(testDataSet, 1)

    else:
        # test
        X_test, y_test = split_sequences(testDataSet, 1)

    n_in = X_test.shape[1] * X_test.shape[2]

    X_test = X_test.reshape((X_test.shape[0], n_in))

    X_test = np.nan_to_num(X_test, nan=month_norm[month])

    return n_in, X_test, y_test
