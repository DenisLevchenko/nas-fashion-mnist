"""
Basic script for creating a .yaml config file.

Easy to edit and add parameters.
The resulting .yaml config file can be used by train_and_test_from_yaml from run_with_lightning.py
or directly by run_with_lightning.py
"""

import yaml

config_dir = 'configs' # directory to put the config file into
config_file_name = 'test_config' # name of the .yaml file created

architecture_type = 'cnn_batch' # see architectures.py for a list of supported architectures

mlp_config = {
    'n_hidden': 3,
    'size_hidden': 16
}

cnn_config = {
    'out_channels' : 64,
    'n_intermediate' : 1,
    'kernel_size' : 3,
    'padding' : 'same',
    'dilation' : 1,
    'dropout_rate' : 0.15
}

net_params = mlp_config if architecture_type=='mlp' else cnn_config

optimizer_config = {
    'lr' : 7.71e-3,
    'weight_decay' : 2.14e-4
}

data_config = {
    'batch_size' : 128,
    'augment' : True
}

checkpoint_config = {
    'monitor' : 'val_accuracy',
    'save_top_k' : 1, # save only the best model
    'mode' : 'max', # max for maximizing, min for minimizaing
    'verbose' : False
}

early_stopping_config = {
    'monitor' : "val_accuracy",
    'min_delta' : 0.00,
    'patience' : 10,
    'mode' : "max",
    'verbose' : False
}

callbacks_config = {
    'checkpoint_config' : checkpoint_config,
    'early_stopping_config' : early_stopping_config
}

lit_module_config = {
    'architecture_type': architecture_type,
    'net_params': net_params,
    'optimizer_params': optimizer_config
}

full_config = {'lit_module_config' : lit_module_config,
               'data_config': data_config,
               'callbacks_config': callbacks_config
}

def save_trial_config_from_study(study, architecture_type: str, trial_number: int | None = None):
    optimizer_keys = ['lr', 'weight_decay'] # for correctly separating optimizer params from the rest
    config_dir = 'configs'
    if trial_number is not None:
        trial = study.trials[trial_number]
        config_name = f'{study.study_name}_trial{trial_number}_config'
    else:
        trial = study.best_trial
        config_name = f'{study.study_name}_best_params'
    variable_params = trial.params.copy()
    set_params = trial.user_attrs.copy()
    all_params = variable_params | set_params
    optimizer_params = {k: all_params[k] for k in optimizer_keys}
    net_params = {k: all_params[k] for k in all_params if k not in optimizer_keys}
    lit_module_config = {'architecture_type': architecture_type,
                         'net_params': net_params,
                         'optimizer_params': optimizer_params}
    full_config = {'lit_module_config': lit_module_config}
    with open(f'{config_dir}/{config_name}.yaml', 'w') as f:
        yaml.safe_dump(full_config, f, sort_keys=False)


with open(f'{config_dir}/{config_file_name}.yaml', 'w') as f:
    yaml.safe_dump(full_config, f, sort_keys=False)