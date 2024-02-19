import os
import pandas as pd
import numpy as np
import wfdb
import ast
from wfdb import processing

import torch
from torch.utils.data import Dataset

class PtbxlDataset(Dataset):
    def __init__(self, ptbxl_path: str, 
                 sampling_rate=100, 
                 transform=None, 
                 target_transform=None, 
                 is_train=True, 
                 use_all=False, 
                 combine_diagnostic=True):
        
        self.dir = ptbxl_path
        ptbxl_sheet = pd.read_csv(self.dir+'ptbxl_database.csv', index_col='ecg_id')
        ptbxl_sheet['scp_codes'] = ptbxl_sheet['scp_codes'].apply(lambda x: ast.literal_eval(x))
        test_fold = 10

        if not use_all:
            if is_train:
                data_mask = ptbxl_sheet['strat_fold'] != test_fold
            else:
                data_mask = ptbxl_sheet['strat_fold'] == test_fold
            ptbxl_sheet = ptbxl_sheet[data_mask]
        
        self.annotations = 'The report of the ECG is that ' + ptbxl_sheet['report'] + '|'
        if combine_diagnostic:
            agg_df = pd.read_csv(self.dir+'scp_statements.csv', index_col=0)
            for col in ['diagnostic', 'form', 'rhythm']:
                df = agg_df[agg_df[col] == 1]
                ptbxl_sheet[col] = ptbxl_sheet['scp_codes'].apply(self._aggregate_diagnostic, df=df)
                self.annotations += col + ': ' + ptbxl_sheet[col].apply(lambda x: ','.join(x))  + '|'

        # 0: Male, 1: Female
        ptbxl_sheet['sex'] = ptbxl_sheet['sex'].replace([0, 1], ['M', 'F'])
        self.annotations = pd.concat((self.annotations, ptbxl_sheet[['age', 'sex']]), axis=1)

        self.sampling_rate = sampling_rate
        if self.sampling_rate == 100:
            self.filenames = ptbxl_sheet['filename_lr']
        else:
            self.filenames = ptbxl_sheet['filename_hr']

        self.transform = transform 
        self.target_transform = target_transform

    def _aggregate_diagnostic(self, y_dic, df):
        tmp = []
        for key in y_dic.keys():
            if key in df.index:
                tmp.append(df.loc[key]['description'])
        return list(set(tmp))

    def __len__(self):
        return len(self.annotations)

    def _resample_and_trim(self, signal_data, target_rate, target_length):
        from scipy import signal

        num_samples = int(len(signal_data) * target_rate / self.sampling_rate)
        resample_signal = signal.resample(signal_data, num_samples)

        start_index = max(0, (len(resample_signal) - target_length) //2)
        return resample_signal[start_index:target_length + start_index]

    def __getitem__(self, idx):
        ecg_path = os.path.join(self.dir, self.filenames.iloc[idx])
        sig, fields = wfdb.rdsamp(ecg_path)
        heart_rate = None
        for lead in range(12):
            xqrs = processing.XQRS(sig=sig[:, lead], fs=fields['fs'])
            xqrs.detect(verbose=False)
            qrs_inds = xqrs.qrs_inds
            if len(qrs_inds) > 1:
                rr_intervals = np.diff(qrs_inds) / fields['fs']
                heart_rate = 60 / np.mean(rr_intervals)
                break
        if heart_rate is None:
            heart_rate = 99999

        if self.sampling_rate == 500:
            x = self._resample_and_trim(sig, 125, 1024)    
        x = torch.as_tensor(x, dtype=torch.float32)

        if self.transform:
            x = self.transform(x)

        info = self.annotations.iloc[idx]
        label = {
            'text': info['report'], 
            'age': info['age'], 
            'gender': info['sex'], 
            'hr': heart_rate 
        }

        if self.target_transform:
            label = self.target_transform(label)
        # x: (L, C)
        return x, label
        
class PtbxlDataset_VAE(Dataset):
    def __init__(self, path: str):
        self.path = path
        self.file_list = os.listdir(path)
        self.file_list.sort(key=lambda x: int(x.split('.')[0]))

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        latent_file = self.file_list[index]
        latent_dict = torch.load(os.path.join(self.path, latent_file))

        return latent_dict['data'], latent_dict['label']


if __name__ == '__main__':
    dataset_path = '/data/0shared/laiyongfan/data_text2ecg/ptb-xl_vae/'
    # data = PtbxlDataset(dataset_path, sampling_rate=500, use_all=True) 
    data = PtbxlDataset_VAE(dataset_path)

    print(data[1000][1])
    from tqdm import tqdm
    for idx, (x, y) in enumerate(tqdm(data)):
        pass
