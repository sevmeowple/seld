import os
import shutil
import numpy as np
import math
import lmdb
from tqdm import tqdm
from collections import defaultdict
# 确保你的环境里能找到这个模块，或者把 datum_pb2.py 放到同级目录
from seld.utils.lmdb_tools.datum_pb2 import SimpleDatum #ty:ignore

def write_list_to_lmdb(file_list, npy_data_dir, npy_label_dir, lmdb_out_dir, segment_length_s=10):
    """
    将指定的文件列表写入 LMDB
    """
    lmdb_map_size = 1099511627776 # 1TB

    if os.path.exists(lmdb_out_dir):
        print(f"Cleaning existing directory: {lmdb_out_dir}")
        shutil.rmtree(lmdb_out_dir)
    os.makedirs(lmdb_out_dir)
    
    print(f"Writing {len(file_list)} files to {lmdb_out_dir} ...")

    env = lmdb.open(lmdb_out_dir, map_size=lmdb_map_size)
    txn = env.begin(write=True)
    lmdb_key_name  = os.path.join(lmdb_out_dir, "keys.txt")
    total_key_file = open(lmdb_key_name, 'w')

    segment_label_frame_num = int(segment_length_s / 0.1)  # 100
    segment_data_frame_num = segment_label_frame_num * 5  # 500

    count = 0
    
    for filename in tqdm(file_list):
        data_path = os.path.join(npy_data_dir, filename)
        label_path = os.path.join(npy_label_dir, filename)

        if not os.path.exists(data_path) or not os.path.exists(label_path):
            print(f"Skipping missing file: {filename}")
            continue

        data = np.load(data_path)
        label = np.load(label_path)
        
        # 确保数据对齐
        label_frame_num = min(data.shape[0]//5, label.shape[0])
        data_frame_num = label_frame_num * 5

        segment_num = math.ceil(data_frame_num / segment_data_frame_num)
        
        for seg_id in range(segment_num):
            # 切片
            segment_data = data[seg_id*segment_data_frame_num:(seg_id+1)*segment_data_frame_num]
            segment_label = label[seg_id*segment_label_frame_num:(seg_id+1)*segment_label_frame_num]
            
            # 生成唯一的 key
            wav_name = filename.split('.')[0] + '_seg_{}_{}'.format(segment_num, seg_id)

            datum = SimpleDatum()
            datum.data = segment_data.astype(np.float32).tobytes()
            datum.label = segment_label.astype(np.float32).tobytes()
            datum.data_dim = segment_data.reshape(segment_data.shape[0], -1).shape[-1]
            datum.label_dim = segment_label.reshape(segment_label.shape[0], -1).shape[-1]
            datum.wave_name = wav_name.encode()
            
            txn.put(wav_name.encode(), datum.SerializeToString())
            total_key_file.write('{}\n'.format(wav_name))
            
            count += 1
            if (count % 1000) == 0:
                txn.commit()
                txn = env.begin(write=True)
                total_key_file.flush()
    
    print(f"Finished. Saved {count} segments to {lmdb_out_dir}")
    txn.commit()
    env.close()
    total_key_file.close()

def split_and_convert(npy_data_dir, npy_label_dir, lmdb_root_dir, split_rule_func, segment_length_s=10):
    """
    主控制函数：扫描目录，根据规则分类，然后分别写入 LMDB
    """
    all_files = [f for f in os.listdir(npy_data_dir) if f.endswith('.npy')]
    files_by_split = defaultdict(list)

    print("Categorizing files...")
    for filename in all_files:
        # 调用规则函数，获取该文件所属的集合名称 (如 'train', 'dev')
        split_name = split_rule_func(filename)
        if split_name:
            files_by_split[split_name].append(filename)
    
    print(f"Split results: { {k: len(v) for k, v in files_by_split.items()} }")

    for split_name, file_list in files_by_split.items():
        # 为每个集合创建一个子目录，例如 lmdb_root/train, lmdb_root/dev
        current_out_dir = os.path.join(lmdb_root_dir, split_name)
        write_list_to_lmdb(
            file_list, 
            npy_data_dir, 
            npy_label_dir, 
            current_out_dir, 
            segment_length_s
        )

# --- 规则定义区 ---

def rule_fold4_is_dev(filename):
    """
    规则：如果是 fold4 则进入 dev，否则进入 train
    """
    # 假设文件名格式包含 'foldX'
    if 'fold4' in filename:
        return 'dev'
    else:
        return 'train'

def rule_custom_example(filename):
    """
    (示例) 更复杂的规则：fold1,2 -> train, fold3 -> val, fold4 -> test
    """
    if 'fold4' in filename:
        return 'test'
    elif 'fold3' in filename:
        return 'val'
    else:
        return 'train'

# ----------------

def debug_read_lmdb(lmdb_dir):
    """
    读取测试，验证写入是否成功
    """
    if not os.path.exists(os.path.join(lmdb_dir, 'keys.txt')):
        print(f"No keys.txt found in {lmdb_dir}, skipping debug.")
        return

    print(f"Debugging reading from {lmdb_dir}...")
    env = lmdb.open(lmdb_dir, readonly=True, readahead=True, lock=False)
    txn = env.begin()
    with open(os.path.join(lmdb_dir, 'keys.txt'), 'r') as f:
        keys = f.readlines()
    
    # 只读前5个看看
    for i, key in enumerate(keys):
        if i >= 5: break 
        k = key.strip().encode()
        val = txn.get(k)
        if val:
            datum = SimpleDatum()
            datum.ParseFromString(val)
            data = np.frombuffer(datum.data, dtype=np.float32).reshape(-1, datum.data_dim)
            label = np.frombuffer(datum.label, dtype=np.float32).reshape(-1, datum.label_dim)
            wav_name = datum.wave_name.decode()
            print(f"  [{i}] Key: {wav_name}, Data: {data.shape}, Label: {label.shape}")
    env.close()

if __name__ == "__main__":
    # 配置路径
    npy_data_dir = '/disk7/zchan/Dataset/Dcase2023/feat_label/foa_dev'
    npy_label_dir = '/disk7/zchan/Dataset/Dcase2023/feat_label/foa_dev_label'
    
    # 输出的根目录，脚本会自动在下面创建 /train 和 /dev 子目录
    lmdb_root_dir = '/disk7/zchan/Dataset/Dcase2023/feat_label/lmdb_foa_dev_split'
    
    # 1. 运行转换
    # 这里传入 rule_fold4_is_dev 函数作为规则
    split_and_convert(
        npy_data_dir, 
        npy_label_dir, 
        lmdb_root_dir, 
        split_rule_func=rule_fold4_is_dev, 
        segment_length_s=10
    )

    # 2. 简单的读取测试
    print("\n--- Testing Read ---")
    debug_read_lmdb(os.path.join(lmdb_root_dir, 'train'))
    debug_read_lmdb(os.path.join(lmdb_root_dir, 'dev'))
