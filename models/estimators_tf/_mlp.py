import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class MLP():
    def __init__(self, input_size, n_layers, n_units, learning_rate, activation, dropout, l2, out_layer, batch_size, epochs, epochs_are_steps=False):
        self.input_size = input_size
        self.n_layers = n_layers
        self.n_units = n_units
        self.learning_rate = learning_rate
        self.activation = activation
        self.dropout = dropout
        self.l2 = l2
        self.out_layer = out_layer
        self.batch_size = batch_size
        self.epochs = epochs
        self.epochs_are_steps = epochs_are_steps

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf.compat.v1.Session()
            with self.sess.as_default():
                self.model = self._build_model()

    def _build_model(self):
        if self.out_layer == 'sigmoid':
            out_act = 'sigmoid'
            loss = keras.losses.BinaryCrossentropy(from_logits=False)
        else:
            out_act = 'linear'
            loss = keras.losses.MeanSquaredError()

        input = layers.Input(shape=(self.input_size,))
        x = layers.Dense(self.n_units, activation=self.activation)(input)
        if self.dropout > 0.0: x = layers.Dropout(self.dropout)(x)

        for _ in range(self.n_layers - 1):
            x = layers.Dense(self.n_units, activation=self.activation)(x)
            if self.dropout > 0.0: x = layers.Dropout(self.dropout)(x)

        kernel_reg = keras.regularizers.l2(self.l2) if self.l2 > 0.0 else None
        x = layers.Dense(1, activation=out_act, kernel_regularizer=kernel_reg)(x)

        model = keras.Model(inputs=input, outputs=x)
        optimiser = keras.optimizers.Adam(self.learning_rate)

        model.compile(optimizer=optimiser, loss=loss)
        return model
    
    def fit(self, X, y):
        bs = self.batch_size if self.batch_size > 0 else X.shape[0]

        if self.epochs_are_steps:
            # Number of steps
            epoch_i = 0
            while(epoch_i < self.epochs):
                for idx_start in range(0, X.shape[0], bs):
                    # Last batch can be smaller
                    idx_end = min(idx_start + bs, X.shape[0])
                    X_batch, y_batch = X[idx_start:idx_end], y[idx_start:idx_end]

                    with self.graph.as_default():
                        with self.sess.as_default():
                            _ = self.model.train_on_batch(X_batch, y_batch)
                    
                    epoch_i += 1
                    # Exit if reached desired number of steps
                    if epoch_i >= self.epochs:
                        break
        else:
            with self.graph.as_default():
                with self.sess.as_default():
                    _ = self.model.fit(X, y, batch_size=bs, epochs=self.epochs, verbose=False)
    
    def predict(self, X):
        # All-purpose predict.
        # Assumes X includes T.
        with self.graph.as_default():
            with self.sess.as_default():
                return self.model.predict(X)

    def predict_proba(self, X):
        # Just for compatibility purposes.
        # (if sigmoid in the output, this returns probabilities)
        return self.predict(X)

    def predict_factual(self, X, t):
        with self.graph.as_default():
            with self.sess.as_default():
                return self.model.predict(np.concatenate([X, t.reshape(-1, 1)], axis=1))
    
    def predict_all(self, X):
        # Assume X excludes T.
        # X = [units, features]
        X0 = np.concatenate([X, np.zeros((X.shape[0], 1))], axis=1)
        X1 = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        with self.graph.as_default():
            with self.sess.as_default():
                y0 = self.model.predict(X0)
                y1 = self.model.predict(X1)
                return y0, y1