"""对比测试: 新旧 SedDoaResult 逻辑一致性"""
import numpy as np
import torch

# 旧版
from seld.utils.result.sed_doa import (
    SedDoaResult,
    SedDoaResult_Streaming_Inf,
    SedDoaResult_Class_Thre,
    SedDoaResult_Streaming_Inf_Class_Thre,
)
# 新版
from seld_v2.metrics.result_collector import SedDoaResultCollector


def make_fake_output(batch_size=2, frames=10, num_classes=13):
    """生成模拟的网络输出 [B, T, 52]"""
    sed = torch.sigmoid(torch.randn(batch_size, frames, num_classes))
    doa = torch.randn(batch_size, frames, num_classes * 3)
    return torch.cat([sed, doa], dim=-1)


def make_fake_wav_names(batch_size=2, segment_length=10):
    """生成模拟的 wav_name, 格式: fold1_room1_mix001_enh1_seg_0_X"""
    return [f"fold1_room1_mix001_enh1_seg_0_{i}" for i in range(batch_size)]


def compare_dicts(d1, d2, label):
    if d1.keys() != d2.keys():
        print(f"  FAIL [{label}]: csv keys differ: {d1.keys()} vs {d2.keys()}")
        return False
    for csv in d1:
        if d1[csv].keys() != d2[csv].keys():
            print(f"  FAIL [{label}]: frame keys differ for {csv}")
            return False
        for frame in d1[csv]:
            if len(d1[csv][frame]) != len(d2[csv][frame]):
                print(f"  FAIL [{label}]: entry count differs at {csv}:{frame}")
                return False
            for e1, e2 in zip(d1[csv][frame], d2[csv][frame]):
                if e1[0] != e2[0] or not np.allclose(e1[1:], e2[1:]):
                    print(f"  FAIL [{label}]: values differ at {csv}:{frame}")
                    return False
    print(f"  PASS [{label}]")
    return True


def test_sed_doa_result():
    """对比 SedDoaResult vs SedDoaResultCollector(segment_length=N, threshold=0.5)"""
    torch.manual_seed(42)
    output = make_fake_output()
    wav_names = make_fake_wav_names()
    seg_len = 10

    old = SedDoaResult(segment_length=seg_len)
    old.add_items(wav_names, output)

    new = SedDoaResultCollector(segment_length=seg_len, threshold=0.5)
    new.add_batch(wav_names, output)

    return compare_dicts(old.get_result(), new.get_result(), "SedDoaResult")


def test_streaming_inf():
    """对比 SedDoaResult_Streaming_Inf vs SedDoaResultCollector(streaming, threshold=0.5)"""
    torch.manual_seed(42)
    output = make_fake_output(batch_size=1)
    csv_name = "fold1_room1_mix001"

    old = SedDoaResult_Streaming_Inf()
    old.add_items(csv_name, output)

    new = SedDoaResultCollector(segment_length=None, threshold=0.5)
    new.add_single(csv_name, output)

    return compare_dicts(old.get_result(), new.get_result(), "Streaming_Inf")


def test_class_thre():
    """对比 SedDoaResult_Class_Thre vs SedDoaResultCollector(segment_length=N, threshold=list)"""
    torch.manual_seed(42)
    output = make_fake_output()
    wav_names = make_fake_wav_names()
    seg_len = 10
    thresholds = [0.7, 0.7, 0.7, 0.45, 0.6, 0.3, 0.65, 0.55, 0.65, 0.7, 0.3, 0.7, 0.7]

    old = SedDoaResult_Class_Thre(segment_length=seg_len)
    old.add_items(wav_names, output)

    new = SedDoaResultCollector(segment_length=seg_len, threshold=thresholds)
    new.add_batch(wav_names, output)

    return compare_dicts(old.get_result(), new.get_result(), "Class_Thre")


def test_streaming_class_thre():
    """对比 SedDoaResult_Streaming_Inf_Class_Thre vs SedDoaResultCollector(streaming, threshold=list)"""
    torch.manual_seed(42)
    output = make_fake_output(batch_size=1)
    csv_name = "fold1_room1_mix001"
    thresholds = [0.7, 0.7, 0.7, 0.45, 0.6, 0.3, 0.65, 0.55, 0.65, 0.7, 0.3, 0.7, 0.7]

    old = SedDoaResult_Streaming_Inf_Class_Thre()
    old.add_items(csv_name, output)

    new = SedDoaResultCollector(segment_length=None, threshold=thresholds)
    new.add_single(csv_name, output)

    return compare_dicts(old.get_result(), new.get_result(), "Streaming_Class_Thre")


if __name__ == "__main__":
    print("对比测试: 新旧 SedDoaResult 逻辑一致性")
    results = [
        test_sed_doa_result(),
        test_streaming_inf(),
        test_class_thre(),
        test_streaming_class_thre(),
    ]
    print(f"\n{'ALL PASSED' if all(results) else 'SOME FAILED'}")
