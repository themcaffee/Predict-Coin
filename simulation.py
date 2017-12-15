import datetime
import time

import os

from database import session, DataPoint
from predict import load_data, compile_model


RETRAIN_BATCH_SIZE = 50


def main():
    pair = 'BTC-USD'
    model_id = int(time.time())
    # Train on data from before simulation start
    end_date = int(datetime.datetime(2017, 10, 1).timestamp())
    X_train, y_train, X_test, y_test = load_data(pair, end_date=end_date)
    model = compile_model()
    model.fit(
        X_train,
        y_train,
        batch_size=512,
        epochs=50,
        verbose=1,
        validation_data=(X_test, y_test)
    )

    # Save the model
    if not os.path.exists("simulation_models"):
        os.mkdir("simulation_models")
    model.save("simulation_models/{}_{}_gdax".format(pair, str(model_id)))

    # Get the the data after simulation start
    datapoints = session.query(DataPoint).filter(
        DataPoint.pair == pair,
        DataPoint.time > end_date
    ).all()

    checkpoint_count = 1
    points_last_batch = 0
    for point in datapoints:
        # todo make decision based on prediction

        if points_last_batch >= RETRAIN_BATCH_SIZE:
            # Retrain model with newer data in n sized batches
            X_train, y_train, X_test, y_test = load_data(pair, end_date=point.time)
            model.fit(
                X_train,
                y_train,
                batch_size=512,
                epochs=1,
                verbose=1,
                validation_date=(X_test, y_test)
            )
            # Save the new model checkpoint
            model.save("simulation_models/{}_{}_{}".format(pair, str(model_id), str(checkpoint_count)))
            score = model.evaluate(X_test, y_test, verbose=0)
            print("New Score at chkpt {}: {}".format(str(checkpoint_count), str(score)))

            points_last_batch = 0
        else:
            points_last_batch += 1


if __name__ == '__main__':
    main()
