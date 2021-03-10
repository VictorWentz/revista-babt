""" Paper Title: Analysis of Artificial Neural Networks Models for
    Forecasting Solar Photovoltaic Generation
    Authors: Wentz VH; Maciel JN; Ledesma JJG; Ando Junior, OH
    Objetive "train.py": Create and train ANN models
    Updated: 08/03/2021
"""


from utils import *
import tensorflow as tf
from tensorflow.keras.backend import clear_session
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.initializers import GlorotNormal
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
import time




def get_model(n_in: int,
              n_out: int,
              nLayers: list,
              dropout: float = 0.15,
              batchNorm: bool = False,
              activ: str = "relu"
              ) -> tf.keras.models.Sequential:
    """[summary]

    Args:
        n_in (int): [description]
        n_out (int): [description]
        nLayers (list): [description]
        dropout (float, optional): [description]. Defaults to 0.15.
        batchNorm (bool, optional): [description]. Defaults to False.
        activ (str, optional): [description]. Defaults to "relu".

    Returns:
        tf.keras.models.Sequential: [description]
    """
    model = Sequential()

    seed = 2459

    if batchNorm:


        for layer in nLayers:
            model.add(Dense(layer,
                            activation=activ,
                            kernel_initializer=GlorotNormal(seed=seed),
                            input_shape=(n_in,)))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))
            n_in = layer

        model.add(Dense(n_out, input_shape=(n_in,)))

        return model

    elif not batchNorm:
        for layer in nLayers:
            model.add(Dense(layer,
                            activation=activ,
                            kernel_initializer=GlorotNormal(seed=seed),
                            input_shape=(n_in,)))
            model.add(Dropout(dropout))
            n_in = layer

        model.add(Dense(n_out, input_shape=(n_in,)))

        return model

def train(minutes: list, layers: list, weatherPath: str, irradiancePath:str,path_to_save_model:str,
          ghi: bool = True, drops: list = None):

    for time_stamp in minutes:
        data = get_data(weatherPath, irradiancePath, timeStamp=time_stamp, drops=drops)

        n_in, X_train, y_train, X_val, y_val, X_test, y_test = data_train_test(data, 2016, ghi)

        for layer in layers:
            if not drops:
                save_model = path_to_save_model + str(time_stamp) + "_minute_model_" + str(layer) + ".h5"
            else:
                save_model = path_to_save_model + str(time_stamp) + "_minute_modelReduced_" + str(layer) + ".h5"


            early = EarlyStopping(monitor="val_loss", patience=10)

            save = ModelCheckpoint(filepath=save_model,
                                   monitor="val_loss",
                                   save_best_only=True)

            mape = tf.keras.metrics.MeanAbsolutePercentageError()
            rmse = tf.keras.metrics.RootMeanSquaredError()

            clear_session()

            model = get_model(n_in, 1, layer)

            model.compile(loss="mse",
                          optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                          metrics=[rmse, mape])

            print(f"[INFO] Train model. Time: {time_stamp} layer: {str(layer)}")
            start = time.time()
            with tf.device('/device:GPU:0'):
                model.fit(
                    X_train,
                    y_train,
                    validation_data=(X_val, y_val),
                    epochs=1000,
                    batch_size=32,
                    verbose=0,
                    callbacks=[early, save])
            end = time.time()

            # print(f"[INFO] Model took: {(end-start)/60} minutes to reach earlyStop")

            # print(f"Results for {str(layer)}")
            # print("--------------------------")
            # print_result(model, X_test, y_test)
            # print("--------------------------")