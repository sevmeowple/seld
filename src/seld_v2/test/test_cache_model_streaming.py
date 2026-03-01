import logging
import os
import time

import joblib
import librosa
import numpy as np
import torch
from tqdm import tqdm

import seld.utils.feature.feature
import seld.utils.feature.parameters as parameters
from config.loader import load_config_generic
from config.schema import StreamingTestConfigFull
from seld_v2.models.resnet_conformer import ResnetConformer
from seld_v2.models.resnet_hoom import HOOM
from seld_v2.data.process import process_foa_input_sed_doa
from seld_v2.metrics.result_collector import SedDoaResultCollector
from seld_v2.training.eval_epoch import save_and_evaluate
from seld_v2.training.experiment import ExperimentDir


def build_model(config: StreamingTestConfigFull) -> torch.nn.Module:
    """Factory: build model from test config."""
    if config.model.name == "hoom":
        if config.streaming.mode == "streaming":
            raise ValueError("HOOM model does not support streaming mode")
        return HOOM(
            in_channel=config.model.in_channel, in_dim=config.model.in_dim,
            out_dim=config.model.out_dim, encoder_dim=config.model.encoder_dim,
            hoom_layout=config.model.hoom_layout,
            ccan_channels=config.model.ccan_channels,
            freq_pool_sizes=config.model.freq_pool_sizes,
        )
    return ResnetConformer(
        in_channel=config.model.in_channel, in_dim=config.model.in_dim,
        out_dim=config.model.out_dim, att_context_size=config.model.att_context_size,
        num_conformer_layer=config.model.num_conformer_layers,
        encoder_dim=config.model.encoder_dim,
        use_dynamic_chunk=config.model.use_dynamic_chunk,
        chunk_candidates=config.model.chunk_candidates,
    )


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_wav_in_sliding_windows(wav_file, segment_length_sec=10, hop_length_sec=10):
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
    logger = exp.setup_logging(log_name="test")
    logger.info(f"Config: {config.model_dump()}")
    logger.info(f"Experiment dir: {exp.root}")

    spec_scaler = joblib.load(str(config.data.norm_file))
    wav_path = str(config.data.wav_folder_path)
    segment_length_sec = config.streaming.segment_length_sec
    hop_length_sec = config.streaming.hop_length_sec
    batch_size = config.data.batch_size
    latency_ms = config.streaming.latency_ms

    params = parameters.get_params()
    dev_feat_cls = seld.utils.feature.feature.FeatureClassO1(params)

    # Initialize model
    model = build_model(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    logger.info(f"Model: {model}")

    if config.model.pre_train:
        model.load_state_dict(torch.load(str(config.model.pre_train_model)))
        logger.info(f"Loaded pre-trained model from {config.model.pre_train_model}")

    model.eval()
    test_result = SedDoaResultCollector()
    total_inference_time = 0
    total_segments = 0

    mode = config.streaming.mode
    overlap_fusion = config.streaming.overlap_fusion

    # Pre-compute frame counts for overlap fusion (offline only)
    # 10s segment -> 500 STFT frames -> 500 ResNet frames -> 100 output frames (MaxPool1d(5))
    output_frames_per_sec = 10  # 100 frames / 10s
    segment_frames = int(segment_length_sec * output_frames_per_sec)
    hop_frames = int(hop_length_sec * output_frames_per_sec)
    overlap_frames = segment_frames - hop_frames  # 0 when no overlap

    for file in tqdm(os.listdir(wav_path)):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        file_path = os.path.join(wav_path, file)
        file_name = os.path.splitext(os.path.basename(file_path))[0]

        if mode == "offline":
            result_array = None
            is_first = True
            for audio_segment, start_time, end_time, is_last_segment in read_wav_in_sliding_windows(
                file_path, segment_length_sec=segment_length_sec, hop_length_sec=hop_length_sec,
            ):
                data = dev_feat_cls.extract_audio_feature(audio_segment)
                data = spec_scaler.transform(data)
                data = process_foa_input_sed_doa(data)
                data = torch.from_numpy(data.astype(np.float32)).to(device).unsqueeze(0)

                start_inference = time.time()
                with torch.no_grad():
                    output = model(data)
                total_inference_time += time.time() - start_inference
                total_segments += 1

                if not overlap_fusion or overlap_frames <= 0:
                    # No overlap: simple concatenation (original behavior)
                    result_array = output if result_array is None else torch.cat([result_array, output], dim=1)
                elif is_first:
                    # First segment: keep first hop_frames + half overlap
                    keep = hop_frames + overlap_frames // 2
                    result_array = output[:, :keep, :]
                    is_first = False
                elif is_last_segment:
                    # Last segment: keep from half overlap onward
                    skip = overlap_frames // 2
                    result_array = torch.cat([result_array, output[:, skip:, :]], dim=1)
                else:
                    # Middle segments: keep middle hop_frames
                    skip = overlap_frames // 2
                    result_array = torch.cat([result_array, output[:, skip:skip + hop_frames, :]], dim=1)

            test_result.add_single(file_name, result_array)
            del result_array
        else:
            resnet_cache = model.get_initial_cache_resnet(batch_size)
            conformer_cache = model.get_initial_cache_conformer(batch_size)
            streaming_outputs = []

            for audio_segment, start_time, end_time, is_last_segment in read_wav_in_sliding_windows(
                file_path, segment_length_sec=segment_length_sec, hop_length_sec=hop_length_sec,
            ):
                data = dev_feat_cls.extract_audio_feature(audio_segment)
                data = spec_scaler.transform(data)
                data = process_foa_input_sed_doa(data)
                data = torch.from_numpy(data.astype(np.float32)).to(device).unsqueeze(0)

                if latency_ms > 0:
                    time.sleep(latency_ms / 1000.0)

                start_inference = time.time()
                with torch.no_grad():
                    output, (resnet_cache, conformer_cache) = model(
                        data, resnet_cache=resnet_cache, conformer_cache=conformer_cache,
                    )
                total_inference_time += time.time() - start_inference
                total_segments += 1
                streaming_outputs.append(output)

            streaming_output = torch.cat(streaming_outputs, dim=1)
            test_result.add_single(file_name, streaming_output)
            del streaming_outputs, streaming_output

    # Log performance
    avg_time = total_inference_time / total_segments if total_segments > 0 else 0
    logger.info(f"Total segments: {total_segments}")
    logger.info(f"Total inference time: {total_inference_time:.4f}s")
    logger.info(f"Avg inference time per segment: {avg_time:.4f}s")
    logger.info(f"Simulated latency: {latency_ms}ms")

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
        "segment_length_sec": segment_length_sec,
    }, seld_metrics)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Streaming Performance Test")
    parser.add_argument("--config", type=str, default="configs/streaming_test.toml")
    parser.add_argument("--latency", type=int, help="Override latency in ms")
    args = parser.parse_args()

    config_kwargs = {"base_path": args.config}
    if args.latency is not None:
        config_kwargs["streaming"] = {"latency_ms": args.latency}

    config = load_config_generic(StreamingTestConfigFull, **config_kwargs)
    main(config)
