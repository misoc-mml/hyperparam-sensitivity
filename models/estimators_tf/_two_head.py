import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import ParameterGrid

from ._common import get_params
from helpers.utils import get_params_df

def _get_ct_data(t, y):
    y_c = np.zeros_like(y)
    w_c = np.zeros_like(y)
    y_c[t < 1] = y[t < 1]
    w_c[t < 1] = 1

    y_t = np.zeros_like(y)
    w_t = np.zeros_like(y)
    y_t[t > 0] = y[t > 0]
    w_t[t > 0] = 1

    return y_c, y_t, w_c, w_t

class TwoHeadSearch():
    def __init__(self, opt):
        self.opt = opt
        self.params_grid = get_params(self.opt.estimation_model)
    
    def run(self, train, test, scaler, iter_id, fold_id):
        X_tr = train[0]
        t_tr = train[1].flatten()
        y_tr = train[2].flatten()
        X_test = test[0]
        t_test = test[1].flatten()

        input_size = X_tr.shape[1]

        y_hats = []
        y0_hats = []
        y1_hats = []
        cate_hats = []
        for params in ParameterGrid(self.params_grid):
            model = TwoHeadNN(input_size, params['n_layers'], params['n_layers2'], params['n_units'], params['learning_rate'], params['activation'], params['dropout'], params['l2'], 'linear', params['batch_size'], params['epochs'], params['epochs_are_steps'])

            model.fit(X_tr, t_tr, y_tr)

            y_hat = model.predict_factual(X_test, t_test)
            y0_hat, y1_hat = model.predict_all(X_test)

            if self.opt.scale_y:
                y0_hat = scaler.inverse_transform(y0_hat)
                y1_hat = scaler.inverse_transform(y1_hat)
            
            cate_hat = y1_hat - y0_hat

            y_hats.append(y_hat)
            y0_hats.append(y0_hat)
            y1_hats.append(y1_hat)
            cate_hats.append(cate_hat)

        if fold_id > 0:
            filename = f'{self.opt.estimation_model}_iter{iter_id}_fold{fold_id}'
        else:
            filename = f'{self.opt.estimation_model}_iter{iter_id}'
        
        y_hats_arr = np.array(y_hats, dtype=object)
        y0_hats_arr = np.array(y0_hats, dtype=object)
        y1_hats_arr = np.array(y1_hats, dtype=object)
        cate_hats_arr = np.array(cate_hats, dtype=object)

        np.savez_compressed(os.path.join(self.opt.output_path, filename), y_hat=y_hats_arr, y0_hat=y0_hats_arr, y1_hat=y1_hats_arr, cate_hat=cate_hats_arr)
        
    def save_params_info(self):
        df_params = get_params_df(self.params_grid)

        df_params.to_csv(os.path.join(self.opt.output_path, f'{self.opt.estimation_model}_params.csv'), index=False)

class TwoHeadNN():
    def __init__(self, input_size, n_layers, n_layers2, n_units, learning_rate, activation, dropout, l2, out_layer, batch_size, epochs, epochs_are_steps=False):
        self.input_size = input_size
        self.n_layers = n_layers
        self.n_layers2 = n_layers2
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
        kernel_reg = keras.regularizers.l2(self.l2) if self.l2 > 0.0 else None

        input = layers.Input(shape=(self.input_size,))
        x = input
        
        x = layers.Dense(self.n_units, activation='linear')(x)
        if self.dropout > 0.0: x = layers.Dropout(self.dropout)(x)
        x = layers.Activation(self.activation)(x)
    
        for _ in range(self.n_layers - 1):
            x = layers.Dense(self.n_units, activation='linear')(x)
            if self.dropout > 0.0: x = layers.Dropout(self.dropout)(x)
            x = layers.Activation(self.activation)(x)

        # Separate hidden layers for the heads.
        if self.n_layers2 > 0:
            n_units2 = int(self.n_units/2)

            c_head = layers.Dense(n_units2, activation='linear')(x)
            if self.dropout > 0.0: c_head = layers.Dropout(self.dropout)(c_head)
            c_head = layers.Activation(self.activation)(c_head)
            for _ in range(self.n_layers2 - 1):
                c_head = layers.Dense(n_units2, activation='linear')(c_head)
                if self.dropout > 0.0: c_head = layers.Dropout(self.dropout)(c_head)
                c_head = layers.Activation(self.activation)(c_head)
            c_head = layers.Dense(1, activation=self.out_layer, kernel_regularizer=kernel_reg, name='c_head')(c_head)

            t_head = layers.Dense(n_units2, activation='linear')(x)
            if self.dropout > 0.0: t_head = layers.Dropout(self.dropout)(t_head)
            t_head = layers.Activation(self.activation)(t_head)
            for _ in range(self.n_layers2 - 1):
                t_head = layers.Dense(n_units2, activation='linear')(t_head)
                if self.dropout > 0.0: t_head = layers.Dropout(self.dropout)(t_head)
                t_head = layers.Activation(self.activation)(t_head)
            t_head = layers.Dense(1, activation=self.out_layer, kernel_regularizer=kernel_reg, name='t_head')(t_head)
        else:
            c_head = layers.Dense(1, activation=self.out_layer, kernel_regularizer=kernel_reg, name='c_head')(x)
            t_head = layers.Dense(1, activation=self.out_layer, kernel_regularizer=kernel_reg, name='t_head')(x)
        
        model = keras.Model(inputs=input, outputs=[c_head, t_head])
        optimiser = keras.optimizers.Adam(self.learning_rate)

        model.compile(optimizer=optimiser, loss=['mse', 'mse'], metrics=None)
        return model
    
    def fit(self, X, t, y):
        bs = self.batch_size if self.batch_size > 0 else len(X)
        y_c, y_t, sw_c, sw_t = _get_ct_data(t, y)

        if self.epochs_are_steps:
            epoch_i = 0
            while(epoch_i < self.epochs):
                for idx_start in range(0, X.shape[0], bs):
                    idx_end = min(idx_start + bs, X.shape[0])
                    X_batch = X[idx_start:idx_end]
                    y_c_batch, y_t_batch = y_c[idx_start:idx_end], y_t[idx_start:idx_end]
                    sw_c_batch, sw_t_batch = sw_c[idx_start:idx_end], sw_t[idx_start:idx_end]

                    with self.graph.as_default():
                        with self.sess.as_default():
                            _ = self.model.train_on_batch(X_batch, (y_c_batch, y_t_batch), sample_weight=(sw_c_batch, sw_t_batch))
                    
                    epoch_i += 1
                    if epoch_i >= self.epochs:
                        break
        else:
            with self.graph.as_default():
                with self.sess.as_default():
                    _ = self.model.fit(X, (y_c, y_t), batch_size=bs, epochs=self.epochs, verbose=False, sample_weight=(sw_c, sw_t))

    def predict_all(self, X):
        with self.graph.as_default():
            with self.sess.as_default():
                preds = self.model.predict(X)
                # [Y0, Y1]
                return preds[0], preds[1]
    
    def predict_factual(self, X, t):
        with self.graph.as_default():
            with self.sess.as_default():
                preds = self.model.predict(X)
        # [n_samples, (y0, y1)]
        ct = np.squeeze(np.array(preds)).T
        # [n_samples, (t0, t1)]
        mask = np.vstack((t==0, t==1)).T
        return ct[mask]