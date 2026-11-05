from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class ModelInferenceConfig:
    root_dir: Path
    model_path: Path
    test_data_path: Path
    output_dir: Path