# DiffuSETS

[![PyTorch](https://img.shields.io/badge/HuggingFace-Model-FFD21E?logo=huggingface)](https://huggingface.co/Laiyf/DiffuSETS)
[![PyTorch](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python)](https://pytorch.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2B-EE4C2C?logo=pytorch)](https://pytorch.org)
[![GitHub repo size](https://img.shields.io/github/repo-size/tuhlnaa/DiffuSETS_Exp-Extended?label=Repo%20size)](https://github.com/tuhlnaa/DiffuSETS_Exp-Extended)

<br>

## Abstract

[![arXiv](https://img.shields.io/badge/arXiv-2501.05932-B31B1B?logo=arxiv)](https://arxiv.org/abs/2501.05932)

Heart disease remains a significant threat to human health. As a non-invasive diagnostic tool, the electrocardiogram (ECG) is one of the most widely used methods for cardiac screening. However, the scarcity of high-quality ECG data, driven by privacy concerns and limited medical resources, creates a pressing need for effective ECG signal generation. Existing approaches for generating ECG signals typically rely on small training datasets, lack comprehensive evaluation frameworks, and overlook potential applications beyond data augmentation. To address these challenges, we propose DiffuSETS, a novel framework capable of generating ECG signals with high semantic alignment and fidelity. DiffuSETS accepts various modalities of clinical text reports and patient-specific information as inputs, enabling the creation of clinically meaningful ECG signals. Additionally, to address the lack of standardized evaluation in ECG generation, we introduce a comprehensive benchmarking methodology to assess the effectiveness of generative models in this domain. Our model achieve excellent results in tests, proving its superiority in the task of ECG generation. Furthermore, we showcase its potential to mitigate data scarcity while exploring novel applications in cardiology education and medical knowledge discovery, highlighting the broader impact of our work.

<p align="center">
    <img width="1000" alt="image" src="https://raw.githubusercontent.com/tuhlnaa/DiffuSETS_Exp-Extended/refs/heads/master/assets/Figure_1.png">
</p>

<br>

## 🚀 Quick Start

```bash
# Create and activate conda environment
conda create -n DiffuSETS python=3.11
conda activate DiffuSETS

# Clone the repository
git clone https://github.com/tuhlnaa/DiffuSETS_Exp-Extended.git
cd DiffuSETS_Exp-Extended

# Install dependencies
# Linux
chmod +x ./script/Install_dependencies.sh
./script/Install_dependencies.sh

# Windows
./script/Install_dependencies.bat
```

<br>

## Inference

Run inference accessing OpenAI api (more flexible):
```sh
python DiffuSETS_inference.py config/all.json
```

For configuration and input settings see this [section](#configurations).

Quick generation using pre-extracted conditions (no api requesting, but make sure you have downloaded prerequisites for inference):
```sh
python -m test_scripts.diversity
```

<br>

## Training

Training script is `DiffuSETS_train.py`, can be launched through 

```sh
python DiffuSETS_train.py config/all.json
```

For configuration settings see this [section](#configurations).

<br>

## Experiment

All scripts concerning with our experiments are included in `./test_scripts/`, please take a look freely.

<br>

## Configurations

**Prerequisities** can be found at [Our Huggingface🤗 Hub](https://huggingface.co/Laiyf/DiffuSETS). Put them under the root of this repo as `./prerequisites/`.

The scripts of `DiffuSETS_train.py` and `DiffuSETS_inference.py` rely on configuration in json format to set up dependency paths and hyper parameters. 

Note that training and inference settings are written in ONE configuration file, so as to ensure the hyper parameters of UNET are the same.

Below is a example of settings with detailed explanation, you can alter it to meet with your environment and design. 

> [!TIP]
> `@key` is comment to `key`, so there is **NO** need to include them in the formal configuration file.

```json
{ 
    "meta": {
        "exp_name": "DiffuSETS", 

        "exp_type": "all", 
        "@exp_type": "the name of experiment saving folders, can use to indicating the model type", 
        
        "condition": true, 
        "@condition": "whether to use patient specific info",

        "vae_latent": true, 
        "@vae_latent": "generating latent or ECG", 

        "device": "cuda:1" 
    }, 

    "dependencies": { 
        "dataset_path": "./prerequisites/mimic_vae_lite.pt", 
        "@dataset_path": "path to mimic_vae.pt (dictdataset) or the folder contains vae latents", 

        "output_dir": "./checkpoints", 

        "vae_path": "./prerequisites/vae_model.pth", 
        "@vae_path": "ALWAYS need to be specified no matter whether generating latent or not", 
    },

    "hyper_para": {
        "epochs": 200, 
        "lr": 5e-4, 
        "batch_size": 512, 
        "num_train_steps": 1000, 
        "unet_kernel_size": 7, 
        "unet_num_level": 7, 
        "beta_start": 0.00085, 
        "beta_end": 0.0120
    }, 

    "@inference_setting": "ONLY concerning with inference scripts", 
    "inference_setting": {
        "inference_timestep": 1000, 

        "gen_batch": 4, 
        "@gen_batch": "number of ECGs generated each time", 

        "save_img": true, 
        "verbose": false,
        "save_path": "./test_sample_all", 
        "unet_path": "./text2ecg/prerequisites/unet_all.pth", 

        "text": "sinus rhythm|abnormal ecg.", 
        "@text": "text reports for generation, multiple reports are split by '|' ", 

        "OPENAI_API_KEY": "",  
        "@OPENAI_API_KEY": "Fill with your API KEY", 

        "age": 50, 
        "hr": 90, 
        "gender": "M"
    }
}
```

<br>

## Citation

Please cite our paper if you use the code, model, or data.

```bibtex
@article{lai2025diffusets,
  title={DiffuSETS: 12-Lead ECG generation conditioned on clinical text reports and patient-specific information},
  author={Lai, Yongfan and Chen, Jiabo and Zhao, Qinghao and Zhang, Deyun and Wang, Yue and Geng, Shijia and Li, Hongyan and Hong, Shenda},
  journal={Patterns},
  year={2025},
  publisher={Elsevier}
}
```
