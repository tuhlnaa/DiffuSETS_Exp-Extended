import argparse
import torch
import os
import json
import numpy as np
from scipy.linalg import sqrtm
from unet.unet_conditional import ECGconditional 
from unet.unet_nocondition import ECGnocondition 
from vae.vae_model import VAE_Decoder
from clip.clip_model import CLIP
from diffusers import DDPMScheduler

def read_features(features_path):
    with open(features_path + '/features.json', 'r') as file:
        features_dict = json.load(file)
        tensor_features = ['text_embed', 'Ori Latent', 'Gen Latent']
        for key in features_dict:
            if key in tensor_features:
                features_dict[key] = eval(features_dict[key])

    return features_dict

@torch.no_grad()
def CLIP_Score_saved_samples(sample_dir:str, clip_model, decoder, device, ptbxl=False):
    """ 
    CLIP Score on saved samples

    sample_dir: path to the sample directory\n
    /path/to/sample_dir
       |-001\n
       |-002\n
       ...\n 
    """
    total_clip_score = 0
    for idx, root in enumerate(os.listdir(sample_dir)):
        feature_dict = read_features(os.path.join(sample_dir, root))
        gen_batch = feature_dict['batch']

        # text_embedding: (gen_B, 1536)
        text_embedding = feature_dict['text_embed']
        if ptbxl:
            text_embedding = torch.tensor(text_embedding).to(device, dtype=torch.float)
        else:
            text_embedding = torch.tensor(text_embedding).squeeze(1).to(device, dtype=torch.float)

        # ecg_latent: (gen_B, 4, 128)
        ecg_latent = feature_dict['Gen Latent']
        ecg_latent = torch.tensor(ecg_latent).to(device)

        # generated ECGs: (gen_B, L, C)
        if decoder: 
            ecgs = decoder(ecg_latent)
        else:
            ecgs = ecg_latent.transpose(2, 1)

        signal_embedding = clip_model.encode_signal(ecgs)

        # signal features: (gen_B, embed_dim)
        signal_features = clip_model.ecg_projector(signal_embedding)
        # text features: (1, embed_dim) -> (gen_B, embed_dim)
        text_features = clip_model.text_projector(text_embedding)
        text_features = text_features.repeat((gen_batch, 1))

        # normalized features
        signal_features = signal_features / signal_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity
        sample_clip_score = torch.trace(signal_features @ text_features.t())

        total_clip_score += sample_clip_score

    total_num = gen_batch * (idx + 1)

    return {'CLIP': total_clip_score.item() / total_num, 'num_samples': total_num}

@torch.no_grad()
def generate_feature_matrix(sample_dir:str, clip_model, device, decoder, use_latent, use_all_batch=True):
    """ 
    Generating feature matrix from experiment folder
    sample_dir: path to the sample directory\n
    /path/to/sample_dir
       |-001\n
       |-002\n
       ...\n 
    use_all_batch: whether to use whole batch, 
    if not, only sample one piece of ecg from each generation folder.\n
    return: dict of `gen` and `real`, which contains feature matrix of shape (num_samples, feature_dim)
    """
    M_gen = []
    M_real = []
    for idx, root in enumerate(os.listdir(sample_dir)):
        feature_dict = read_features(os.path.join(sample_dir, root))

        # ecg_latent: (gen_B, 4, 128)
        gen_latent = feature_dict['Gen Latent']
        gen_latent = torch.tensor(gen_latent).to(device)

        # generated ECGs: (gen_B, L, C)
        if use_latent: 
            gen_ecgs = decoder(gen_latent)
        else: 
            gen_ecgs = gen_latent.transpose(2, 1)
        
        # gen_ecg_features: (gen_B, feature_dim) or (1, feature_dim)
        gen_ecg_embedding = clip_model.encode_signal(gen_ecgs)
        gen_ecg_features = clip_model.ecg_projector(gen_ecg_embedding)

        if not use_all_batch:
            gen_ecg_features = gen_ecg_features[0].unsqueeze(0)
        gen_ecg_features = gen_ecg_features.cpu()

        M_gen.append(gen_ecg_features)

        # ori_latent: (1, 4, 128)
        ori_latent = feature_dict['Ori Latent']
        ori_latent = torch.tensor(ori_latent).to(device)
        ori_ecgs = decoder(ori_latent)
        ori_ecg_embedding = clip_model.encode_signal(ori_ecgs)
        # ori_ecg_feature: (1, feature_dim)
        ori_ecg_features = clip_model.ecg_projector(ori_ecg_embedding)
        ori_ecg_features = ori_ecg_features.cpu()
        M_real.append(ori_ecg_features)

    M_gen = torch.concat(M_gen)
    M_real = torch.concat(M_real)

    # M_gen: (num_samples, num_features)
    return {'gen': M_gen, 'real': M_real}


def FID_score(M1: torch.Tensor, M2: torch.Tensor):
    M1, M2 = M1.numpy(), M2.numpy()
    mu1, sigma1 = M1.mean(axis=0), np.cov(M1, rowvar=False)
    mu2, sigma2 = M2.mean(axis=0), np.cov(M2, rowvar=False)

    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

class ManifoldDetector():
    def __init__(self, data: torch.Tensor, k=3):
        self.k = k
        self.data = data

        # Compute pairwise distances
        distances = torch.sqrt(torch.sum((self.data.unsqueeze(1) - self.data.unsqueeze(0))**2, dim=2))

        # Get indices of k-nearest neighbors
        _, indices = torch.topk(distances, k=self.k + 1, dim=1, largest=False)
        indices = indices[:, 1:]  # Exclude the point itself

        # Compute radius as the distance to the k-th nearest neighbor
        self.radii = torch.gather(distances, 1, indices[:, -1].view(-1, 1))

def is_in_manifold(test_point: torch.Tensor, manifold_detector: ManifoldDetector):
    distances = torch.sqrt(torch.sum((manifold_detector.data - test_point)**2, dim=1))
    is_inside = distances <= manifold_detector.radii.squeeze()
    return is_inside.any()

def points_in_manifold(test_points: torch.Tensor, manifold_detector: ManifoldDetector):
    count = 0
    for point in test_points:
       count += is_in_manifold(point, manifold_detector) 

    return count

def precision_recall(M_g, M_r, k=3):
    """ 
    Compute the precision and recall value for generation result\n
    M_g: feature matrix of generated ECG\n
    M_r: feature matrix of real ECG\n
    k: using distance from k nearest neighborhood to constuct manifold 
    """
    manifold_detector_g = ManifoldDetector(M_g, k)
    manifold_detector_r = ManifoldDetector(M_r, k)

    state = {}
    num_precision = points_in_manifold(M_g, manifold_detector_r)
    state['precision'] = num_precision / M_g.shape[0]

    num_recall = points_in_manifold(M_r, manifold_detector_g)
    state['recall'] = num_recall / M_r.shape[0] 

    state['F1'] = 2 / (1.0 / state['recall'] + 1.0 / state['precision']) 

    return state

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="A simple way to manage experiments"
    )

    # Add arguments
    parser.add_argument(
        "--exp_type", type=str, required=True,
        help="experimetn type in ['pvc', 'pac', ...]"
    )
    parser.add_argument(
        "--gpu_ids", type=int, default=0,
        help="gpu index"
    )
    args = parser.parse_args()

    exp_type = args.exp_type

    save_path = f'./exp/disease/{exp_type}'
    device_str = f"cuda:{args.gpu_ids}"
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    
    # CLIP model
    clip_model_root = './prerequistes/clip_model.pth'
    clip_model = CLIP(embed_dim=64)
    clip_model_weight = torch.load(clip_model_root, map_location=device)
    clip_model.load_state_dict(clip_model_weight)
    clip_model.eval()
    clip_model = clip_model.to(device)

    # UNET
    n_channels = 4 
    num_train_steps = 1000
    n_inference_steps = 1000 

    # VAE
    decoder = VAE_Decoder()
    # VAE_path
    vae_path = './prerequisites/vae_model.pth'
    checkpoint = torch.load(vae_path, map_location=device)
    decoder.load_state_dict(checkpoint['decoder'])
    decoder.eval()
    decoder = decoder.to(device)

    result = CLIP_Score_saved_samples(sample_dir=save_path,
                                      clip_model=clip_model, 
                                      decoder=decoder,
                                      device=device)

    state = generate_feature_matrix(sample_dir=save_path, clip_model=clip_model, device=device, decoder=decoder, use_latent=True, use_all_batch=True)
    M_gen, M_real = state['gen'], state['real']
    fid_score = FID_score(M_real, M_gen) 
    num_samples = M_real.shape[0]
    scaler = FID_score(M_real[:num_samples // 2], M_real[num_samples // 2:])
    r_FID = fid_score / scaler
    result['FID'] = fid_score
    result['rFID'] = r_FID
    result_1 = precision_recall(M_g=M_gen, M_r=M_real)
    result.update(result_1)

    result_line = f'{exp_type}\t'
    
    for key in result.keys():
        if key in ['num_samples']:
            continue
        result_line += f'{key}:{result[key]:.3f} '
    print(result_line)