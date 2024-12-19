import torch 
import tqdm 


def parse_label_normal(text: str): 
    if 'normal ecg' in text.lower():
        return 1
    else:
        return 0 

def parse_label_af(text: str): 
    if 'atrial fibrillation' in text.lower():
        return 0
    else:
        return 1 

def parse_label_pvc(text: str): 
    text = text.lower()
    if 'pvc' in text or 'ventricular premature' in text or 'premature ventricular' in text:
        return 0
    else:
        return 1 

def build_dataset(src: str, dst: str): 
    assert src != dst 
    src_dataset = torch.load(src)  

    train_dict = dict()
    valid_dict = dict()
    test_dict = dict()
    for key, value in tqdm.tqdm(src_dataset.items()): 
        text = value['label']['text'] 
        data = value['data']
        label = parse_label_pvc(text) 

        value = { 'label': {'label': label, 'text': text}, 
                 'data': data}
        
        if key < 40000:
            train_dict[key] = value 
        elif key < 45000:
            valid_dict[key] = value 
        else:
            test_dict[key] = value 
    
    torch.save(train_dict, dst + 'train.pt') 
    torch.save(valid_dict, dst + 'valid.pt') 
    torch.save(test_dict, dst + 'test.pt') 

if __name__ == '__main__': 
    src = './prerequisites/mimic_vae_lite_0.pt'
    dst = './prerequisites/mimic_vae_clf_pvc_'
    build_dataset(src, dst)