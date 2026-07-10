import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_slice, plot_parallel_coordinate, plot_contour
import sys
import logging
import plotly
from dataclasses import dataclass

studies_dir = "optuna_databases"
study_name = "cnn_64"
storage_name = f"sqlite:///{studies_dir}/{study_name}.db"
study = optuna.load_study(study_name=study_name, storage=storage_name)

print(study.best_params)

plot_optimization_history(study)

# plot_contour(study, params=["n_hidden", "size_hidden"])

plot_slice(study)

plot_parallel_coordinate(study)