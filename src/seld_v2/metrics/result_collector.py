from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch


class SedDoaResultCollector:
    """
    SED+DOA 结果收集器，合并原始4个变体。

    原始类对应的参数组合:
        SedDoaResult:                          segment_length=N,    threshold=0.5
        SedDoaResult_Class_Thre:               segment_length=N,    threshold=[0.7, 0.7, ...]
        SedDoaResult_Streaming_Inf:            segment_length=None, threshold=0.5
        SedDoaResult_Streaming_Inf_Class_Thre: segment_length=None, threshold=[0.7, 0.7, ...]
    """

    def __init__(
        self,
        segment_length: int | None = None,
        threshold: float | List[float] = 0.5,
        num_classes: int = 13,
    ):
        self.segment_length = segment_length
        self.threshold = threshold
        self.num_classes = num_classes
        self.output_dict: Dict[str, Dict[int, list]] = {}

    def _get_threshold(self, class_idx: int) -> float:
        if isinstance(self.threshold, list):
            return self.threshold[class_idx]
        return self.threshold

    @staticmethod
    def _to_numpy(t: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(t, torch.Tensor):
            return t.detach().cpu().numpy()
        return t

    def _add_item(self, csv_name: str, start_frame: int, sed_pred: np.ndarray, doa_pred: np.ndarray) -> None:
        if csv_name not in self.output_dict:
            self.output_dict[csv_name] = {}
        n = self.num_classes
        for frame_cnt in range(sed_pred.shape[0]):
            out_frame = frame_cnt + start_frame
            for class_cnt in range(sed_pred.shape[1]):
                if sed_pred[frame_cnt][class_cnt] > self._get_threshold(class_cnt):
                    if out_frame not in self.output_dict[csv_name]:
                        self.output_dict[csv_name][out_frame] = []
                    self.output_dict[csv_name][out_frame].append([
                        class_cnt,
                        doa_pred[frame_cnt][class_cnt],
                        doa_pred[frame_cnt][class_cnt + n],
                        doa_pred[frame_cnt][class_cnt + 2 * n],
                    ])

    def add_batch(self, wav_names: List[str], net_output: torch.Tensor | np.ndarray) -> None:
        """非 streaming 批量添加，从 wav_name 解析 csv_name 和 start_frame"""
        n = self.num_classes
        sed = self._to_numpy(net_output[:, :, :n])
        doa = self._to_numpy(net_output[:, :, n:n * 4])
        for b, wav_name in enumerate(wav_names):
            items = wav_name.split('_')
            csv_name = '_'.join(items[:-3])
            start_frame = int(items[-1]) * self.segment_length  # type: ignore[operator]
            self._add_item(csv_name, start_frame, sed[b], doa[b])

    def add_single(self, csv_name: str, net_output: torch.Tensor | np.ndarray) -> None:
        """streaming 模式添加单条，squeeze batch 维度"""
        n = self.num_classes
        sed = self._to_numpy(net_output[:, :, :n]).squeeze(0)
        doa = self._to_numpy(net_output[:, :, n:n * 4]).squeeze(0)
        self._add_item(csv_name, 0, sed, doa)

    def get_result(self) -> Dict[str, Dict[int, list]]:
        return self.output_dict
