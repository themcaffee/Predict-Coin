import os
import time

import matplotlib
import numpy as np
from keras.layers import LSTM, Dropout, Dense, Activation
from keras.models import Sequential

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

from database import session, DataPoint


def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    #Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()


def plot_results_full(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()


def load_data(pair):
    seq_len = 51
    # Get all of the closing data from the db
    datapoints = session.query(DataPoint).filter_by(pair=pair).all()
    data = []
    for point in datapoints:
        data.append(point.close)
    result = []
    for index in range(len(data) - seq_len):
        result.append(data[index: index + seq_len])

    # Normalize the data
    result = normalize_data(result)
    result = np.array(result)

    # Break into test/train sets
    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]
    # TODO Why is this here?
    np.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return [x_train, y_train, x_test, y_test]


def normalize_data(window_data):
    normalised_data = []
    for window in window_data:
        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data


def compile_model():
    model = Sequential()

    model.add(LSTM(
        input_dim=1,
        output_dim=50,
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        100,
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=1))
    model.add(Activation('linear'))

    start = time.time()
    model.compile(loss='mse', optimizer='rmsprop')
    print('compilation time : ', time.time() - start)
    return model


def predict_point_by_point(model, data):
    #Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted


def predict_sequence_full(model, data, window_size):
    #Shift the window by 1 new prediction each time, re-run predictions on new window
    curr_frame = data[0]
    predicted = []
    for i in range(len(data)):
        predicted.append(model.predict(curr_frame[np.newaxis, :, :])[0, 0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
    return predicted


def predict_sequences_multiple(model, data, window_size, prediction_len):
    #Predict sequence of 50 steps before shifting prediction run forward by 50 steps
    prediction_seqs = []
    for i in range(round(len(data)/prediction_len) - 1):
        curr_frame = data[i*prediction_len]
        predicted = []
        for j in range(prediction_len):
            predicted.append(model.predict(curr_frame[np.newaxis, :, :])[0, 0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs


def main():
    pair = 'BTC-USD'
    # Get the data
    X_train, y_train, X_test, y_test = load_data(pair)

    # Compile the model
    model = compile_model()

    # Train the model
    model.fit(
        X_train,
        y_train,
        batch_size=512,
        epochs=50,
        verbose=1,
        validation_data=(X_test, y_test)
    )

    # Save the model
    if not os.path.exists("saved_models"):
        os.mkdir('saved_models')

    model.save('saved_models/{}_{}_gdax'.format(pair, str(time.time())))

    score = model.evaluate(X_test, y_test, verbose=0)
    print("Score: {}".format(str(score)))

    # Plot the predictions
    print('Building predictions for multiple different windows...')
    predictions_multiple = predict_sequences_multiple(model, X_test, 50, 50)
    plot_results_multiple(predictions_multiple, y_test, 50)

    print('Building predictions for a full run...')
    predictions_full = predict_sequence_full(model, X_test, 50)
    plot_results_full(predictions_full, y_test)

    print('Building predictions for point by point...')
    predictions_point = predict_point_by_point(model, X_test)
    plot_results_full(predictions_point, y_test)


if __name__ == '__main__':
    main()
