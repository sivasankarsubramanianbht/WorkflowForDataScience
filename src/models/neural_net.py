# --- neural_net.py ---
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

def build_nn(input_dim, n_units_1=64, n_units_2=32):
    model = Sequential([
        Dense(n_units_1, activation='relu', input_shape=(input_dim,)),
        Dense(n_units_2, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_final_model(X_train, y_train, input_dim, params):
    model = build_nn(input_dim, params['n_units_1'], params['n_units_2'])
    model.fit(
        X_train, y_train,
        epochs=params['epochs'],
        batch_size=params['batch_size'],
        verbose=0,
        callbacks=[EarlyStopping(monitor='val_mae', patience=3, restore_best_weights=True)]
    )
    return model