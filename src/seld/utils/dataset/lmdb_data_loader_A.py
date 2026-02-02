import os
import numpy as np
import lmdb
import joblib

import torch
from torch.utils.data import Dataset
#from visual_src.visual_tools import VisualTools
from seld.utils.lmdb_tools.datum_pb2 import SimpleDatum #ty:ignore

class LmdbDataset(Dataset):
    def __init__(self, lmdb_dir, split, normalized_features_wts_file=None, ignore=None, segment_len=None, data_process_fn=None) -> None:
        super().__init__()
        self.split = split
        self.ignore = ignore
        self.segment_len = segment_len
        self.data_process_fn = data_process_fn
        #self.visial_tools = VisualTools()
        self.keys = []
        with open(os.path.join(lmdb_dir, 'keys.txt'), 'r') as f:
            lines = f.readlines()
            for k in lines:
                if self.ignore is not None and self.ignore in k:
                    continue
                if int(k[4]) in self.split: # check which split the file belongs to
                    self.keys.append(k.strip())
        self.lmdb_dir = str(lmdb_dir)
        self.env = None
        self.spec_scaler = None
        if normalized_features_wts_file is not None:
            self.spec_scaler = joblib.load(normalized_features_wts_file)
        
    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        if self.env is None:
            self.env = lmdb.open(self.lmdb_dir, readonly=True, readahead=True, lock=False)
        txn = self.env.begin()
        with txn.cursor() as cursor:
            k = self.keys[index].strip().encode()
            cursor.set_key(k)
            datum=SimpleDatum()
            datum.ParseFromString(cursor.value())
            data = np.frombuffer(datum.data, dtype=np.float32).reshape(-1, datum.data_dim)
            if self.spec_scaler is not None:
                data = self.spec_scaler.transform(data)
            #pdb.set_trace()
            label = np.frombuffer(datum.label, dtype=np.float32).reshape(-1, datum.label_dim)

            wav_name = datum.wave_name.decode()
            if self.segment_len is not None and label.shape[0] < self.segment_len:
                data = np.pad(data, pad_width=((0,self.segment_len*5-data.shape[0]), (0,0)))
                label = np.pad(label, pad_width=((0,self.segment_len-label.shape[0]), (0,0)))
            if self.data_process_fn is not None:
                data, label = self.data_process_fn(data, label)

            #print('feat {}'.format(data.shape))
            #print('label {}'.format(label.shape))
            #print('wavname {}'.format(wav_name))
        return {'data': data, 'label':label, 'wav_name':wav_name}


    def collater(self, samples):
        feats = [s['data'] for s in samples]
        labels = [s['label'] for s in samples]
        wav_names = [s['wav_name'] for s in samples]

        collated_feats = np.stack(feats, axis=0)
        collated_labels = np.stack(labels, axis=0)

        out = {}
        out['input'] = torch.from_numpy(collated_feats)
        out['target'] = torch.from_numpy(collated_labels)
        out['wav_names'] = wav_names

        return out

class LmdbDataset_Pad(Dataset):
    def __init__(self, lmdb_dir, split, normalized_features_wts_file=None, ignore=None,
                  segment_len= None, data_process_fn=None)-> None:
        super().__init__()
        self.split = split
        self.ignore = ignore
        self.segment_len=segment_len
        self.data_process_fn= data_process_fn
        self.keys =[]
        with open(os.path.join(lmdb_dir, 'keys.txt'),'r') as f:
            lines = f.readlines()
            for k in lines:
                if self.ignore is not None and self.ignore in k:
                    continue
                if int(k[4])in self.split: # check which split the file belongs to
                    self.keys.append(k.strip())
        self.lmdb_dir = str(lmdb_dir)
        self.env = None
        self.spec_scaler = None
        if normalized_features_wts_file is not None:
            self.spec_scaler = joblib.load(normalized_features_wts_file)

    def __len__(self):
        return len(self.keys)
    
    def __getitem__ (self,index):
        if self.env is None:
            self.env = lmdb.open(self.lmdb_dir, readonly=True, readahead=True, lock=False)
        txn = self.env.begin()
        with txn.cursor() as cursor:       
            k= self.keys[index].strip().encode()
            cursor.set_key(k)
            datum=SimpleDatum()
            datum.ParseFromString(cursor.value())
            data = np.frombuffer(datum.data, dtype=np.float32).reshape(-1, datum.data_dim)
            if self.spec_scaler is not None:
                data = self.spec_scaler.transform(data)
            label = np.frombuffer(datum.label, dtype=np.float32).reshape(-1, datum.label_dim)

            wav_name = datum.wave_name.decode()
            pad_width =0 
            if self.segment_len is not None and label.shape[0]< self.segment_len:
                pad_width=self.segment_len - label.shape[0]
                data = np.pad(data, pad_width=((0,self.segment_len*5-data.shape[0]),(0,0)))
                label = np.pad(label, pad_width=((0,self.segment_len-label.shape[0]),(0,0)))
            if self.data_process_fn is not None:
                data, label=self.data_process_fn(data, label)
        return {'data': data, 'label':label, 'wav_name':wav_name, 'pad_width': pad_width}
    
    def collater(self,samples):
        feats =[s['data'] for s in samples]
        labels =[s['label']for s in samples]
        wav_names =[s['wav_name'] for s in samples]
        pad_width =[s['pad_width'] for s in samples]

        collated_feats =np.stack(feats, axis=0)
        collated_labels=np.stack(labels,axis=0)
        collated_pad_width = np.stack(pad_width, axis=0)
    
        out ={}
        out['input']= torch.from_numpy(collated_feats)
        out['target']= torch.from_numpy(collated_labels)
        out['wav_names']= wav_names
        out['pad_width']= torch.from_numpy(collated_pad_width)
        return out
