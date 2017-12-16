import datetime
import time

import os
from pprint import pprint

import keras
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from database import session, DataPoint
from predict import load_data, compile_model, predict_point_by_point, plot_results_full, predict_sequences_multiple, \
    plot_results_multiple

RETRAIN_BATCH_SIZE = 5


def main():
    pair = 'BTC-USD'
    epochs = 1

    # Train on data from before simulation start
    original_end_date = int(datetime.datetime(2017, 10, 1).timestamp())
    X_train, y_train, X_test, y_test, scalerX, scalerY = load_data(pair, end_date=original_end_date)

    # Load this model if it has already been run
    model_filename = "simulation_models/{}_{}_{}_gdax".format(pair, str(original_end_date), str(epochs))
    if os.path.exists(model_filename):
        model = keras.models.load_model(model_filename)
    else:
        model = compile_model()
        model.fit(
            X_train,
            y_train,
            batch_size=512,
            epochs=epochs,
            verbose=1,
            validation_data=(X_test, y_test)
        )

        # Save the model
        if not os.path.exists("simulation_models"):
            os.mkdir("simulation_models")

        model.save("simulation_models/{}_{}_{}_gdax".format(pair, str(original_end_date), str(epochs)))

    # Get the the data after simulation start
    datapoints = session.query(DataPoint).filter(
        DataPoint.pair == pair,
        DataPoint.time > original_end_date
    ).all()


    # Go through the simulation
    checkpoint_count = 1
    points_last_batch = 0
    data = []
    seq_len = 51
    for point in datapoints:
        data.append(point.close)
        if len(data) < 100:
            continue

        result = []
        for index in range(len(data) - seq_len):
            result.append(data[index: index + seq_len])

        # Normalize the data
        result = np.array(result)

        # Do some shit
        x_test = result[0:, :-1]
        y_test = result[0:, -1]

        # Normalize
        x_test = scalerX.transform(x_test)
        y_test = scalerY.transform(y_test)

        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        # Denormalize and show prediction
        y_test = scalerY.inverse_transform(y_test)

        predicted = predict_point_by_point(model, x_test, denormalize_scaler=scalerY)
        plot_results_full(predicted, y_test)

        # predictions_multiple = predict_sequences_multiple(model, x_test, 50, 50, denormalize_scaler=scalerY)
        # plot_results_multiple(predictions_multiple, y_test, 50)
        # todo make decision based on prediction

        if points_last_batch >= RETRAIN_BATCH_SIZE:
            # Retrain model with newer data in n sized batches
            X_train, y_train, X_test, y_test, scalerX, scalerY = load_data(pair, end_date=point.time)
            model.fit(
                X_train,
                y_train,
                batch_size=512,
                epochs=1,
                verbose=1,
                validation_data=(X_test, y_test)
            )
            # Save the new model checkpoint
            model.save("simulation_models/{}_{}_{}_gdax_{}".format(pair, str(original_end_date), str(epochs), str(checkpoint_count)))
            score = model.evaluate(X_test, y_test, verbose=0)
            print("New Score at chkpt {}: {}".format(str(checkpoint_count), str(score)))

            points_last_batch = 0
        else:
            points_last_batch += 1


if __name__ == '__main__':
    main()
