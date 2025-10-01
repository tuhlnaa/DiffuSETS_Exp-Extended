import json
import ecg_plot
import torch
import numpy as np

from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
from matplotlib import pyplot as plt

# Import custom modules
from utils.text_to_emb import prompt_propcess


def get_embedding_from_api(text: str, openai_key: str) -> np.ndarray: 
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

    # Directly create batched tensor: (embedding_dim, ) -> (batch_size, 1, embedding_dim)
    embedding_tensor = torch.from_numpy(embedding).unsqueeze(0).unsqueeze(0)
    embedding_tensor = embedding_tensor.expand(batch_size, 1, -1).to(device)

    return embedding_tensor


def prepare_condition_dict(gender: str, age: float, hr: float, batch_size: int, 
                           device: torch.device) -> dict[str, torch.Tensor]:
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


def generation_from_net(diffused_model, net, batch_size: int, device: torch.device, text_embed: torch.Tensor, 
                        condition: dict[str, torch.Tensor] | None = None, num_channels: int = 4, dim: int = 128) -> torch.Tensor:
    """Generate samples using diffusion model."""
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


def save_ecg_images(gen_ecg: torch.Tensor, save_path: Path, sample_rate: float = 102.4, 
                    save_signal: bool = True, save_image: bool = True) -> None:
    """
    Save generated ECG as raw signals and/or images to disk.
    
    Args:
        gen_ecg: Generated ECG tensor of shape (batch_size, seq_len, num_leads)
        save_path: Directory path to save outputs
        sample_rate: Sampling rate in Hz (default: 102.4)
        save_signal: Whether to save raw signal as .npy file
        save_image: Whether to save ECG plot as .png image
    """
    for j in range(len(gen_ecg)):
        output = gen_ecg[j].squeeze(0).detach().cpu().numpy()
        output = output.T

        # Save raw signal as numpy array
        if save_signal:
            signal_path = save_path / f'{j}_ecg_signal.npy'
            np.save(signal_path, output)
        
        # Save ECG plot as image
        if save_image:
            ecg_plot.plot(output, sample_rate)
            plt.savefig(save_path / f'{j}_ecg_image.png', dpi=200)
            plt.close()


def batch_generate_ECG(settings: dict, unet, diffused_model, decoder, condition: bool) -> None:
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
    save_path = Path(settings['save_path'])
    save_path.mkdir(parents=True, exist_ok=True)
    
    save_img = settings['save_img']
    save_signal = settings['save_signal']
    
    if not save_img and not save_signal:
        print("Warning: Both save_img and save_signal are False. No output will be saved.")

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
            device
        )
        metadata.update({
            "gender": settings['gender'],
            "age": settings['age'],
            "heart rate": settings['hr']
        })

    # Move models to device
    unet.to(device)
    decoder.to(device)
    unet.eval()
    decoder.eval()

    # Generate latent representations
    latent = generation_from_net(
        diffused_model, 
        unet, 
        batch_size, 
        device, 
        text_embed=text_embed, 
        condition=condition_dict,
        num_channels=4,
        dim=128
    )

    # Generate and save ECG outputs if requested
    if save_img or save_signal:
        with torch.no_grad():
            gen_ecg = decoder(latent)
        
        save_ecg_images(gen_ecg, save_path, sample_rate=102.4, 
                        save_signal=save_signal, save_image=save_img)
        if verbose:
            print(f"Generated ECG shape: {gen_ecg.shape}, dtype: {gen_ecg.dtype}")

        print(f"Saved {batch_size} ECG samples to {save_path}")

    # Save generation metadata to JSON file
    filepath = save_path / 'features.json'
    with open(filepath, 'w') as json_file:
        json.dump(metadata, json_file, indent=4)


def batch_generate_ECG_novae(settings: dict, unet, diffused_model, condition: bool) -> None:
    """
    Generate batch of ECG signals using diffusion model (NoVAE version).
    
    Args:
        settings: Dictionary containing generation parameters
        unet: U-Net model for noise prediction
        diffused_model: Diffusion model for sampling
        condition: Whether to use conditional generation
    """
    # Extract settings
    save_path = Path(settings['save_path'])
    save_path.mkdir(parents=True, exist_ok=True)
    
    save_img = settings['save_img']
    save_signal = settings['save_signal']
    
    if not save_img and not save_signal:
        print("Warning: Both save_img and save_signal are False. No output will be saved.")

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
            device
        )
        metadata.update({
            "gender": settings['gender'],
            "age": settings['age'],
            "heart rate": settings['hr']
        })

    # Move model to device
    unet.to(device)
    unet.eval()

    # Generate ECG in latent space: (B, C, L)
    gen_ecg = generation_from_net(
        diffused_model, 
        unet, 
        batch_size, 
        device, 
        text_embed=text_embed, 
        condition=condition_dict,
        num_channels=12,
        dim=1024
    )

    # (batch_size, num_leads, seq_len) -> (batch_size, seq_len, num_leads)
    gen_ecg = gen_ecg.transpose(-1, -2)

    # Save ECG images if requested
    if save_img or save_signal:
        save_ecg_images(gen_ecg, save_path, sample_rate=102.4, 
                        save_signal=save_signal, save_image=save_img)
        if verbose:
            print(f"Generated ECG shape: {gen_ecg.shape}, dtype: {gen_ecg.dtype}")

        print(f"Saved {batch_size} ECG samples to {save_path}")

    # Save generation metadata to JSON file
    filepath = save_path / 'features.json'
    with open(filepath, 'w') as json_file:
        json.dump(metadata, json_file, indent=4)
