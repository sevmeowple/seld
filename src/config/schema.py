from pathlib import Path
from typing import List, Literal

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    name: Literal["resnet", "transformer", "hoom", "dhoom", "resnet_conformer_dhoom"] = "resnet"
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
    sample_chunks_from_candidates: bool = Field(
        default=False,
        description="If True, sample directly from chunk_candidates list; "
        "If False, use U2-style: 50%% full context, 50%% uniform [1, max(chunk_candidates)]",
    )
    num_conformer_layers: int = Field(default=6, gt=0)
    encoder_dim: int = Field(default=256, gt=0)
    hoom_layout: List[str] = Field(
        default=["ccan", "ccan", "buan", "buan"],
        description="Block layout for HOOM model",
    )
    ccan_channels: List[int] = Field(
        default=[64, 128],
        description="Output channels for each CCAN block",
    )
    freq_pool_sizes: List[int] = Field(
        default=[2, 2],
        description="Frequency pooling size after each CCAN block",
    )
    num_mhsa: int = Field(default=0, description="Number of MHSA layers after conformer (DHOOM)")
    streaming_loss_weight: float = Field(default=1.0, description="Weight for streaming loss (DHOOM)")
    early_stop_head: Literal["offline", "streaming", "average", "none"] = Field(
        default="offline", description="Which head to use for early stopping (DHOOM)",
    )
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
    scheduler: Literal["tristage", "warmup"] = Field(
        default="tristage",
        description="Learning rate scheduler type: tristage (SpecAugment style) or warmup (WeNet U2 style)",
    )
    warmup_steps: int = Field(
        default=25000, gt=0,
        description="Number of warmup steps for warmup scheduler (ignored for tristage)",
    )
    # TriStage scheduler parameters (ignored when using warmup scheduler)
    init_lr_scale: float = Field(default=0.01, description="Initial LR as fraction of peak LR (tristage only)")
    final_lr_scale: float = Field(default=0.05, description="Final LR as fraction of peak LR (tristage only)")
    warmup_ratio: float = Field(default=0.1, description="Fraction of steps for warmup (tristage only)")
    hold_ratio: float = Field(default=0.6, description="Fraction of steps for hold (tristage only)")


class ResultConfig(BaseModel):
    log_output_path: Path
    log_interval: int = Field(default=100, gt=0)
    checkpoint_output_dir: Path
    dcase_output_dir: Path


class StreamingTestConfig(BaseModel):
    """Configuration for streaming performance tests"""

    mode: Literal["offline", "streaming", "streaming_chunk_mask", "streaming_cache"] = Field(
        default="streaming", description="Test mode: offline, streaming (alias for chunk_mask), streaming_chunk_mask (chunk-based attention), or streaming_cache (true streaming with cache)"
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
