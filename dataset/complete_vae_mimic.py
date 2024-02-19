import os
import torch
import pandas as pd
from tqdm import tqdm
from dataset.mimic_iv_ecg_dataset import VAE_MIMIC_IV_ECG_Dataset

def complete_patient_info(target_path: str):
    assert 'backup' not in target_path, f"Carefully check target path: {target_path} !"
    patient_table_path = '/data1_science/1shared/physionet.org/files/mimiciv/2.2/hosp/patients.csv'
    patient_table = pd.read_csv(patient_table_path, index_col='subject_id', low_memory=False)

    # note_table_path = '/data1_science/1shared/physionet.org/files/mimic-iv-note/2.2/note/discharge.csv'
    # note_table = pd.read_csv(note_table_path, low_memory=False)
    # print(note_table.head())
    # print(note_table[note_table['note_id'] == '10000084-DS-17'].text.tolist()[0])
    # print(note_table.info())

    vae_path = '/data/0shared/laiyongfan/data_text2ecg/mimic_vae_backup'
    data = VAE_MIMIC_IV_ECG_Dataset(vae_path)

    exculde_list = []
    for idx, (latent, label) in enumerate(tqdm(data)):
        if label['hr'] > 99998:
            exculde_list.append(idx)
            continue

        patient_id = label['subject_id']
        query = patient_table[patient_table.index == patient_id] 

        # mimic iv note do not release
        # note_id = label['note_id']
        # label['text'] = note_table[note_table['note_id'] == note_id]['text'].to_list()[0]

        try:
            label['age'] = query['anchor_age'].to_list()[0]
            label['gender'] = query['gender'].to_list()[0]

        except IndexError:
            exculde_list.append(idx)
            continue

        # save_dict = {
        #     'data': latent, 
        #     'label': label
        # }
        # torch.save(save_dict, os.path.join(target_path, f'{idx}.pt'))

    print(len(exculde_list))
    with open('exclude_list.txt', 'w') as f:
        f.writelines('\n'.join(list(map(str, exculde_list)))) 

if __name__ == '__main__':
    target_path = '/data/0shared/laiyongfan/data_text2ecg/mimic_vae'
    complete_patient_info(target_path=target_path)
