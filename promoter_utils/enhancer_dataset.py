import torch
import copy
import pickle
import os

class EnhancerDataset(torch.utils.data.Dataset):
    def __init__(self, name, data_dir, split='train'):
        if name == 'Mel':
            all_data = pickle.load(open(os.path.join(data_dir, 'DeepMEL2_data.pkl'), 'rb'))
        else:
            all_data = pickle.load(open(os.path.join(data_dir, 'DeepFlyBrain_data.pkl'), 'rb'))
        self.seqs = torch.argmax(torch.from_numpy(copy.deepcopy(all_data[f'{split}_data'])), dim=-1)
        self.clss = torch.argmax(torch.from_numpy(copy.deepcopy(all_data[f'y_{split}'])), dim=-1)
        self.num_cls = all_data[f'y_{split}'].shape[-1]
        self.alphabet_size = 4

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx], self.clss[idx]