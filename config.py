import yaml

architecture_type = 'cnn_batch'

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

with open("config.yaml", "w") as f:
    yaml.safe_dump(full_config, f, sort_keys=False)