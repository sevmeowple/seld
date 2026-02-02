from pathlib import Path
from typing import Literal, List

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    name: Literal["resnet", "transformer"] = "resnet"
    in_channel: int = Field(default=256, gt=0)
    in_dim: int = Field(default=256, gt=0)
    out_dim: int = Field(default=256, gt=0)
    att_context_size: List[int] = Field(default=[100, 49], min_items=2)
    num_conformer_layers: int = Field(default=6, gt=0)
    encoder_dim: int = Field(default=256, gt=0)
    pre_train:bool = Field(default=False, description="Whether to load pre-trained model")
    pre_train_model:Path
class DataConfig(BaseModel):
    train_lmdb_dir: Path
    test_lmdb_dir: Path
    ref_files_dir: Path
    norm_file: Path
    segment_len: int = Field(default=100, gt=0)
    select_num: int = Field(default=1000, gt=0) #video per seconds
    batch_size: int = Field(default=32, gt=0)
    train_ignore: str = Field(default="enh", min_length=1) # No ACS
    test_ignore: str = Field(default="enh", min_length=1)

class TrainConfig(BaseModel):
    lr: float = 1e-3
    train_num_workers: int = Field(default=4, gt=0)
    test_num_workers: int = Field(default=4, gt=0)
    nb_steps: int = Field(default=1000, gt=0)
    seed: int = 42

class ResultConfig(BaseModel):
    log_output_path: Path
    log_interval: int = Field(default=100, gt=0)
    checkpoint_output_dir: Path
    dcase_output_dir: Path

class Config(BaseModel):
    model: ModelConfig
    data: DataConfig
    train: TrainConfig
    result: ResultConfig
    debug: bool = False
