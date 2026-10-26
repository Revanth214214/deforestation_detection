from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_url: str
    local_data_file: Path

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    data_file:Path
    STATUS_FILE: Path

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    model_name: str
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    pos_weight: float
    encoder_name: str
    in_channels: int
    classes: int