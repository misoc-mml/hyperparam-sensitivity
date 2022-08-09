from ._mlp import MLP

def get_params(name):
    if name == 'mlp':
        return {'n_layers': [1, 2], 'n_units': [4, 8, 16, 32, 64, 128], 'learning_rate': [1e-4, 1e-3], 'activation': ['relu'], 'dropout': [0.25], 'l2': [1e-2], 'batch_size': [32, 64, 128, -1], 'epochs': [10000], 'epochs_are_steps': [True]}
    elif name == 'two-head':
        return {'n_layers': [1, 2], 'n_layers2': [0, 1, 2], 'n_units': [4, 8, 16, 32, 64, 128], 'learning_rate': [1e-4, 1e-3], 'activation': ['relu'], 'dropout': [0.25], 'l2': [1e-2], 'batch_size': [32, 64, 128, -1], 'epochs': [10000], 'epochs_are_steps': [True]}
    else:
        raise ValueError("Unrecognised 'get_params' key.")

def get_mlp_reg(input_size, params):
    return MLP(input_size, params['n_layers'], params['n_units'], params['learning_rate'], params['activation'], params['dropout'], params['l2'], 'linear', params['batch_size'], params['epochs'], params['epochs_are_steps'])

def get_mlp_clf(input_size, params):
    return MLP(input_size, params['n_layers'], params['n_units'], params['learning_rate'], params['activation'], params['dropout'], params['l2'], 'sigmoid', params['batch_size'], params['epochs'], params['epochs_are_steps'])