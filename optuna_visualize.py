import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_slice, plot_parallel_coordinate, plot_contour
import sys
import logging
import plotly
from dataclasses import dataclass

architecture_type = "mlp"
# architecture_type = "cnn"


study_name = f"{architecture_type}_optimization_hidden_n_and_size"
storage_name = f"sqlite:///{study_name}.db"
study = optuna.load_study(study_name=study_name, storage=storage_name)

print(study.best_params)

plot_optimization_history(study)

plot_contour(study, params=["n_hidden", "size_hidden"])

plot_slice(study, params=["n_hidden", "size_hidden"])

plot_parallel_coordinate(study, params=["n_hidden", "size_hidden"])