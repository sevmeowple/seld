from seld_v2.data.dataset import LmdbDataset

lmdb_dir = "/disk7/zchan/Dataset/Dcase2023/dev/train_lmdb"
dataset = LmdbDataset(lmdb_dir, split=[1], return_pad_width=False)
sample = dataset[0]

print("data shape:", sample['data'].shape)
print("label shape:", sample['label'].shape)
print("label dim:", sample['label'].shape[1])
print("label dim == 52?", sample['label'].shape[1] == 52)