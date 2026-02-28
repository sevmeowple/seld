from pathlib import Path
from typing import List, Literal

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    name: Literal["resnet", "transformer", "hoom"] = "resnet"
    in_channel: int = Field(default=256, gt=0)
    in_dim: int = Field(default=256, gt=0)
    out_dim: int = Field(default=256, gt=0)
    att_context_size: List[int] = Field(default=[100, 49], min_items=2)
    use_dynamic_chunk: bool = Field(
        default=True, description="Use dynamic chunk training"
    )
    chunk_candidates: List[int] = Field(
        default=[4, 9, 24, 49],
        description="Candidate chunk sizes for dynamic chunk training",
    )
    num_conformer_layers: int = Field(default=6, gt=0)
    encoder_dim: int = Field(default=256, gt=0)
    num_hoom_layers: int = Field(default=8, gt=0)
    hoom_fusion: Literal["concat", "add"] = "concat"
    pre_train: bool = Field(
        default=False, description="Whether to load pre-trained model"
    )
    pre_train_model: Path


class DataConfig(BaseModel):
    train_lmdb_dir: Path
    test_lmdb_dir: Path
    ref_files_dir: Path
    norm_file: Path
    segment_len: int = Field(default=100, gt=0)
    select_num: int = Field(default=1000, gt=0)  # video per seconds
    batch_size: int = Field(default=32, gt=0)
    train_ignore: str = Field(default="enh", min_length=1)  # No ACS
    test_ignore: str = Field(default="enh", min_length=1)


class StreamingTestDataConfig(BaseModel):
    wav_folder_path: Path
    ref_files_dir: Path
    norm_file: Path
    batch_size: int = Field(default=1, gt=0)


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


class StreamingTestConfig(BaseModel):
    """Configuration for streaming performance tests"""

    mode: Literal["streaming", "offline"] = Field(
        default="streaming", description="Test mode: streaming or offline"
    )
    latency_ms: int = Field(
        default=0, ge=0, description="Simulated latency in milliseconds"
    )
    segment_length_sec: float = Field(
        default=1.0, gt=0, description="Segment length in seconds"
    )
    hop_length_sec: float = Field(
        default=1.0, gt=0, description="Hop length in seconds"
    )
    overlap_fusion: bool = Field(
        default=True,
        description="When hop < segment, crop boundary frames and keep only middle portion for offline",
    )


class Config(BaseModel):
    exp_name: str = Field(
        default="unnamed", description="Experiment name for output directory"
    )
    model: ModelConfig
    data: DataConfig
    train: TrainConfig
    result: ResultConfig
    debug: bool = False


class StreamingTestConfigFull(BaseModel):
    """Full configuration for streaming tests"""

    exp_name: str = Field(
        default="unnamed", description="Experiment name for output directory"
    )
    model: ModelConfig
    data: StreamingTestDataConfig
    streaming: StreamingTestConfig
    debug: bool = False
