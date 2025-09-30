import os
import json
import ecg_plot
import torch
import numpy as np

from tqdm import tqdm
from openai import OpenAI
from matplotlib import pyplot as plt

from utils.text_to_emb import prompt_propcess


def get_embedding_from_api(text: str, openai_key: str): 
    """Get text embedding from OpenAI API."""
    text = prompt_propcess(text) 

    client = OpenAI(api_key=openai_key)

    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )

    return np.array(response.data[0].embedding)


def prepare_text_embedding(text: str, openai_key: str, batch_size: int, device: torch.device) -> torch.Tensor:
    """Prepare text embedding tensor for batch processing."""
    embedding = get_embedding_from_api(text, openai_key)
    embedding = embedding.astype(np.float32)

    # Directly create batched tensor: (embedding_dim, ) ->(batch_size, 1, embedding_dim)
    embedding_tensor = torch.from_numpy(embedding).unsqueeze(0).unsqueeze(0)
    embedding_tensor = embedding_tensor.expand(batch_size, 1, -1).to(device)

    return embedding_tensor


def prepare_condition_dict(gender: str, age: float, hr: float, batch_size: int, 
                           device: torch.device):
    """Prepare condition dictionary for conditional generation."""
    condition_dict = {
        'gender': 1 if gender == 'M' else 0,
        'age': age,
        'heart rate': hr
    }
    
    # Convert to batched tensors: (batch_size, 1, 1)
    for key, value in condition_dict.items():
        tensor = torch.full((batch_size, 1, 1), value, dtype=torch.float32).to(device)
        condition_dict[key] = tensor

    return condition_dict


def generation_from_net(diffused_model, net, batch_size, device, text_embed, 
                        condition=None, num_channels=4, dim=128):
    """Generate samples using diffusion model."""
    net.eval()
    xi = torch.randn(batch_size, num_channels, dim).to(device)
    
    timesteps = tqdm(diffused_model.timesteps)
    for i in timesteps:
        t = torch.full((batch_size,), i, dtype=torch.long).to(device)
        
        with torch.no_grad():
            # change this line to fit your unet 
            if condition:
                noise_predict = net(xi, t, text_embed, condition)
            else:
                noise_predict = net(xi, t, text_embed)

            xi = diffused_model.step(
                model_output=noise_predict, 
                timestep=i, 
                sample=xi
            )['prev_sample']
    
    return xi 


def save_ecg_images(gen_ecg: torch.Tensor, save_path: str, sample_rate: float = 102.4):
    """Save generated ECG images to disk."""
    for j in range(len(gen_ecg)):
        output = gen_ecg[j].squeeze(0).detach().cpu().numpy()
        ecg_plot.plot(output.T, sample_rate)            
        plt.savefig(os.path.join(save_path, f'{j} Generated ECG.png'))
        plt.close()


def save_metadata(save_path: str, metadata: dict):
    """Save generation metadata to JSON file."""
    filepath = os.path.join(save_path, 'features.json')
    with open(filepath, 'w') as json_file:
        json.dump(metadata, json_file, indent=4)
    print(f"Features successfully written to {filepath}")


def batch_generate_ECG(settings: dict, unet, diffused_model, decoder, condition: bool):
    """
    Generate batch of ECG signals using diffusion model.
    
    Args:
        settings: Dictionary containing generation parameters
        unet: U-Net model for noise prediction
        diffused_model: Diffusion model for sampling
        decoder: Decoder model for latent-to-signal conversion
        condition: Whether to use conditional generation
    """
    # Extract settings
    save_path = settings['save_path']
    os.makedirs(save_path, exist_ok=True)
    
    save_img = settings['save_img']
    if not save_img:
        print("Skipping image generation...")


    batch_size = settings['gen_batch']
    device = torch.device(settings['device'] if torch.cuda.is_available() else "cpu")
    verbose = settings['verbose']
    
    # Prepare text embedding
    text_embed = prepare_text_embedding(
        settings['text'], 
        settings['OPENAI_API_KEY'], 
        batch_size, 
        device
    )

    if verbose:
        print(f"Text embedding shape: {text_embed.shape}")
    
    # Prepare metadata
    metadata = {
        "batch": batch_size,
        "Diagnosis": settings['text']
    }

    # Prepare conditional inputs if needed
    condition_dict = None
    if condition:
        condition_dict = prepare_condition_dict(
            settings['gender'], 
            settings['age'], 
            settings['hr'], 
            batch_size, 
            device, 
        )
        metadata.update({
            "gender": settings['gender'],
            "age": settings['age'],
            "heart rate": settings['hr']
        })

    # Move models to device
    unet.to(device)
    decoder.to(device)

    # Generate latent representations
    latent = generation_from_net(
        diffused_model, 
        unet, 
        batch_size, 
        device, 
        text_embed=text_embed, 
        condition=condition_dict
    )

    # Generate and save images if requested
    if save_img:
        gen_ecg = decoder(latent)
        save_ecg_images(gen_ecg, save_path)

    # Save metadata
    save_metadata(save_path, metadata)    

