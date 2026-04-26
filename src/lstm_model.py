import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout

def build_model(input_shape):
    model = Sequential()

    model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape))
    model.add(Dropout(0.2))

    model.add(Bidirectional(LSTM(32)))
    model.add(Dropout(0.2))

    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_lstm(X_train, y_train, epochs, batch_size):
    model = build_model((X_train.shape[1], 1))

    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    os.makedirs("models", exist_ok=True)
    model.save("models/lstm.h5")

    return model