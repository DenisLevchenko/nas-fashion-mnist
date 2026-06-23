

mlp_config = {
    'n_hidden': 3,
    'size_hidden': 16
}

cnn_config = {
    'out_channels' : 32,
    'n_intermediate' : 1,
    'kernel_size' : 3,
    'padding' : 'same',
    'dilation' : 1,
    'dropout_rate' : 0.15
}

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

early_stop_config = {
    'monitor' : "val_accuracy",
    'min_delta' : 0.00,
    'patience' : 10,
    'mode' : "max",
    'verbose' : False
}