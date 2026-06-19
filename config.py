from dataclasses import dataclass, asdict

@dataclass
class MLPConfig:
    n_hidden: int = 3
    size_hidden: int = 16


@dataclass
class CNNConfig:
    out_channels: int = 51
    n_intermediate: int = 2
    kernel_size: int = 3
    padding: str = 'same'
    dilation: int = 1
    dropout_rate: float = 0.21


@dataclass
class TrainConfig:
    learning_rate: float = 7.71e-4
    weight_decay: float = 2.14e-4
    batch_size: int = 128


@dataclass
class SetupConfig:
    architecture_type: str = 'cnn_rich' # mlp_basic, mlp, cnn_basic, cnn, cnn_rich, cnn_expand
    augment: bool = True