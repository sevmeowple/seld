"""Test script for ResNetConformerDHOOM model.

Supports both offline and streaming modes with proper chunk handling.
Streaming mode uses chunk-by-chunk processing (not full sequence with cache).
"""

import logging
import os
import time
from pathlib import Path

import joblib
import librosa
import numpy as np
import torch
from tqdm import tqdm

import seld.utils.feature.feature
import seld.utils.feature.parameters as parameters
from config.loader import load_config_generic
from config.schema import StreamingTestConfigFull
from seld_v2.models.resnet_conformer_dhoom import ResNetConformerDHOOM
from seld_v2.data.process import process_foa_input_sed_doa
from seld_v2.metrics.result_collector import SedDoaResultCollector
from seld_v2.training.eval_epoch import save_and_evaluate
from seld_v2.training.experiment import ExperimentDir


def build_model(config: StreamingTestConfigFull) -> ResNetConformerDHOOM:
    """Build ResNetConformerDHOOM model from config."""
    return ResNetConformerDHOOM(
        in_channel=config.model.in_channel,
        in_dim=config.model.in_dim,
        out_dim=config.model.out_dim,
        num_conformer_layers=config.model.num_conformer_layers,
        encoder_dim=config.model.encoder_dim,
        num_mhsa=config.model.num_mhsa,
        att_context_size=config.model.att_context_size,
        use_dynamic_chunk=config.model.use_dynamic_chunk,
        chunk_candidates=config.model.chunk_candidates,
        sample_chunks_from_candidates=config.model.sample_chunks_from_candidates,
    )


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def extract_features(audio_segment, dev_feat_cls, spec_scaler):
    """Extract and normalize features from audio segment."""
    data = dev_feat_cls.extract_audio_feature(audio_segment)
    data = spec_scaler.transform(data)
    data = process_foa_input_sed_doa(data)
    return data


def inference_offline(model, data, device):
    """Offline inference: process full sequence, use offline head output."""
    data = torch.from_numpy(data.astype(np.float32)).to(device).unsqueeze(0)
    with torch.no_grad():
        model_output = model(data)
        # DHOOM returns (offline_pred, online_pred) tuple
        if isinstance(model_output, tuple):
            output = model_output[0]  # Use offline head
        else:
            output = model_output
    return output


def inference_streaming_chunk_mask(model, data, device, chunk_size):
    """Streaming inference using chunk mask (decoding_chunk_size).

    This mode uses the model's internal _build_chunk_mask to restrict attention.
    Good for: simulating streaming behavior while processing full segments.
    """
    data = torch.from_numpy(data.astype(np.float32)).to(device).unsqueeze(0)
    with torch.no_grad():
        # Use decoding_chunk_size to enable chunk-based attention
        _, online_pred = model(data, decoding_chunk_size=chunk_size)
    return online_pred


def inference_streaming_cache(model, data, device, chunk_size, batch_size):
    """Streaming inference using cache (true streaming).

    This mode processes audio chunk-by-chunk with cache state.
    Good for: measuring actual streaming latency and memory usage.
    """
    resnet_cache = model.get_initial_cache_resnet(batch_size)
    conformer_cache = model.get_initial_cache_conformer(batch_size)

    total_frames = data.shape[0]  # T frames
    outputs = []

    with torch.no_grad():
        for start in range(0, total_frames, chunk_size):
            end = min(start + chunk_size, total_frames)
            chunk = data[start:end]  # (chunk_T, F)

            # Pad last chunk if needed
            if end - start < chunk_size:
                pad_len = chunk_size - (end - start)
                chunk = np.pad(chunk, ((0, pad_len), (0, 0)), mode='constant')

            chunk_tensor = torch.from_numpy(chunk.astype(np.float32)).to(device).unsqueeze(0)
            output, (resnet_cache, conformer_cache) = model(
                chunk_tensor, resnet_cache=resnet_cache, conformer_cache=conformer_cache
            )
            outputs.append(output)

    return torch.cat(outputs, dim=1)


def read_wav_in_sliding_windows(wav_file, segment_length_sec=10, hop_length_sec=10):
    """Read wav file in sliding windows."""
    audio, sr = librosa.load(wav_file, sr=None, mono=False)
    segment_length_samples = int(segment_length_sec * sr)
    hop_length_samples = int(hop_length_sec * sr)
    num_channels = audio.shape[0]
    audio_length_samples = audio.shape[1]

    for start_sample in range(0, audio_length_samples, hop_length_samples):
        end_sample = start_sample + segment_length_samples
        is_last_segment = start_sample + hop_length_samples >= audio_length_samples

        if end_sample > audio_length_samples:
            padded = np.zeros((num_channels, segment_length_samples))
            padded[:, :audio_length_samples - start_sample] = audio[:, start_sample:]
            audio_segment = padded
        else:
            audio_segment = audio[:, start_sample:end_sample]

        start_time = start_sample / sr
        end_time = min(end_sample, audio_length_samples) / sr
        yield audio_segment, start_time, end_time, is_last_segment

        if is_last_segment:
            break


def main(config: StreamingTestConfigFull):
    exp = ExperimentDir(base_dir="experiments", name=config.exp_name)
    logger = exp.setup_logging(log_name="test_dhoom")
    logger.info(f"Config: {config.model_dump()}")
    logger.info(f"Experiment dir: {exp.root}")

    spec_scaler = joblib.load(str(config.data.norm_file))
    wav_path = str(config.data.wav_folder_path)
    segment_length_sec = config.streaming.segment_length_sec
    hop_length_sec = config.streaming.hop_length_sec
    batch_size = config.data.batch_size

    params = parameters.get_params()
    dev_feat_cls = seld.utils.feature.feature.FeatureClassO1(params)

    # Initialize model
    model = build_model(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    logger.info(f"Model: {model.__class__.__name__}")

    if config.model.pre_train:
        checkpoint_path = Path(config.model.pre_train_model)
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        model.load_state_dict(torch.load(str(checkpoint_path), map_location=device))
        logger.info(f"Loaded checkpoint from {checkpoint_path}")

    model.eval()

    mode = config.streaming.mode
    test_result = SedDoaResultCollector()
    total_inference_time = 0
    total_segments = 0

    # Determine chunk size for streaming
    att_context_size = config.model.att_context_size
    if len(att_context_size) >= 2 and att_context_size[1] > 0:
        chunk_size = att_context_size[1] + 1  # e.g., 49 + 1 = 50
    else:
        chunk_size = 50  # default
    logger.info(f"Using chunk_size: {chunk_size}")

    for file in tqdm(sorted(os.listdir(wav_path))):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        file_path = os.path.join(wav_path, file)
        file_name = os.path.splitext(os.path.basename(file_path))[0]

        result_array = None

        for audio_segment, start_time, end_time, is_last_segment in read_wav_in_sliding_windows(
            file_path, segment_length_sec=segment_length_sec, hop_length_sec=hop_length_sec,
        ):
            data = extract_features(audio_segment, dev_feat_cls, spec_scaler)

            start_inference = time.time()

            if mode == "offline":
                output = inference_offline(model, data, device)
            elif mode in ["streaming", "streaming_chunk_mask"]:
                # Use chunk mask (decoding_chunk_size) for streaming simulation
                # "streaming" is alias for "streaming_chunk_mask" (backward compatible)
                output = inference_streaming_chunk_mask(model, data, device, chunk_size)
            elif mode == "streaming_cache":
                # True streaming with cache, chunk-by-chunk processing
                output = inference_streaming_cache(model, data, device, chunk_size, batch_size)
            else:
                raise ValueError(f"Unknown mode: {mode}. Use 'offline', 'streaming' (alias for chunk_mask), 'streaming_chunk_mask', or 'streaming_cache'")

            total_inference_time += time.time() - start_inference
            total_segments += 1

            result_array = output if result_array is None else torch.cat([result_array, output], dim=1)

        test_result.add_single(file_name, result_array)
        del result_array

    # Log performance
    avg_time = total_inference_time / total_segments if total_segments > 0 else 0
    logger.info(f"Total segments: {total_segments}")
    logger.info(f"Total inference time: {total_inference_time:.4f}s")
    logger.info(f"Avg inference time per segment: {avg_time:.4f}s")

    # Save & evaluate
    seld_metrics = save_and_evaluate(
        test_result.get_result(),
        str(exp.dcase_output_dir),
        str(config.data.ref_files_dir),
    )
    print(f"ER/F/LE/LR/SELD: {seld_metrics['ER']:.4f}/{seld_metrics['F']:.4f}/"
          f"{seld_metrics['LE']:.4f}/{seld_metrics['LR']:.4f}/{seld_metrics['seld_scr']:.4f}")

    exp.log_metrics("test", {
        "mode": mode,
        "att_context_size": str(config.model.att_context_size),
        "chunk_size": chunk_size,
        "segment_length_sec": segment_length_sec,
    }, seld_metrics)

    logger.info("Test completed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("DHOOM Model Test")
    parser.add_argument("--config", type=str, required=True, help="Path to config TOML file")
    parser.add_argument("--mode", type=str, choices=["offline", "streaming", "streaming_chunk_mask", "streaming_cache"],
                        help="Override mode in config. 'streaming' is alias for 'streaming_chunk_mask'")
    args = parser.parse_args()

    config_kwargs = {"base_path": args.config}
    if args.mode:
        config_kwargs["streaming"] = {"mode": args.mode}

    config = load_config_generic(StreamingTestConfigFull, **config_kwargs)
    main(config)
