"""
从 LMDB 数据重新生成 StandardScaler 归一化文件
"""
import os
import lmdb
import numpy as np
import joblib
from sklearn import preprocessing
from tqdm import tqdm

from seld.utils.lmdb_tools.datum_pb2 import SimpleDatum #ty:ignore


def regenerate_scaler_from_lmdb(lmdb_dir, output_file):
    """
    从 LMDB 数据库重新生成 StandardScaler
    
    Args:
        lmdb_dir: LMDB 数据目录
        output_file: 输出的归一化权重文件路径
    """
    print(f"Reading from LMDB: {lmdb_dir}")
    print(f"Output scaler file: {output_file}")
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 读取所有 keys
    keys_file = os.path.join(lmdb_dir, 'keys.txt')
    if not os.path.exists(keys_file):
        raise FileNotFoundError(f"keys.txt not found in {lmdb_dir}")
    
    with open(keys_file, 'r') as f:
        keys = [line.strip() for line in f.readlines()]
    
    print(f"Found {len(keys)} samples in LMDB")
    
    # 打开 LMDB 环境
    env = lmdb.open(lmdb_dir, readonly=True, readahead=True, lock=False)
    
    # 创建 StandardScaler
    spec_scaler = preprocessing.StandardScaler()
    
    # 遍历所有数据，增量拟合
    print("Fitting StandardScaler...")
    with env.begin() as txn:
        for key in tqdm(keys):
            key_bytes = key.encode()
            value = txn.get(key_bytes)
            
            if value is None:
                print(f"Warning: Key {key} not found in LMDB")
                continue
            
            # 解析数据
            datum = SimpleDatum()
            datum.ParseFromString(value)
            data = np.frombuffer(datum.data, dtype=np.float32).reshape(-1, datum.data_dim)
            
            # 增量拟合
            spec_scaler.partial_fit(data)
    
    env.close()
    
    # 保存归一化权重文件
    joblib.dump(spec_scaler, output_file)
    print(f"StandardScaler saved to: {output_file}")
    print(f"  Mean shape: {spec_scaler.mean_.shape}")
    print(f"  Scale shape: {spec_scaler.scale_.shape}")
    print("Done!")


if __name__ == "__main__":
    # 配置路径
    train_lmdb_dir = "/disk7/zchan/Dataset/Dcase2023/dev/train_lmdb"
    output_scaler_file = "/disk7/zchan/Dataset/Dcase2023/dev/norm_dir/24h_ACS_foa_wts_fromlmdb"
    
    # 重新生成归一化文件
    regenerate_scaler_from_lmdb(train_lmdb_dir, output_scaler_file)
