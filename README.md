# DiffuSETS

Note: All the python commands below should be run at current directory i.e. `/Path/to/DiffuSETS/`

## Inference 

To run inference:

```sh
python DiffuSETS_inference.py path/to/config.json
```

For configuration settings see this [section](#configurations).

For prequisities and pretrained UNET model, you can find them [here](https://huggingface.co/Laiyf/DiffuSETS). Just put them in `./prerequisites/` if you want to use them.

## Training

Training script is `DiffuSETS_train.py`, can be launched through 

```sh
python DiffuSETS_train.py path/to/config.json
```

For configuration settings see this [section](#configurations).

For prequisities, you can find them [here](https://huggingface.co/Laiyf/DiffuSETS). Just put them in `./prerequisites/` if you want to use them.

The training scripts can take in two kinds of preprocessed dataset, dictionary format and single item slices style, while we offer the dictionary [here](https://huggingface.co/Laiyf/DiffuSETS). Further introduction can be seen in this [section](#dataset). 


## Configurations

The scripts of `DiffuSETS_train.py` and `DiffuSETS_inference.py` rely on configuration in json format to set up dependency paths and hyper parameters. 

Note that training and inference settings are written in ONE configuration file, so as to ensure the hyper parameters of UNET are the same.

Below is a example of settings in `test.json` with detailed explanation , you can alter it to meet with your environment and design. Reminder: `@key` is comment to `key`, so there is no need to be included in the formal configuration file.

```json
{ 
    "meta": {
        "exp_name": "test", 

        "exp_type": "unet", 
        "@exp_type": "the name of experiment saving folders, i.e. checkpoints/unet_1/", 

        "condition": true, 
        "@condition": "whether to use patient specific info",

        "vae_latent": true, 
        "@vae_latent": "generating latent or ECG", 

        "device": "cuda:1", 

        "use_dictdataset": true,  
        "@use_dictdataset": "whether to use dictdataset"
    }, 

    "dependencies": { 
        "dataset_path": "./prerequisites/mimic_vae_lite.pt", 
        "@dataset_path": "path to mimic_vae.pt (dictdataset) or the folder contains vae latents", 

        "checkpoints_dir": "./checkpoints", 

        "vae_path": "./text2ecg/prerequisites/vae_model.pth", 
        "@vae_path": "ALWAYS need to be specified no matter whether generating latent or not", 

        "text_embed_path": "./text2ecg/prerequisites/mimic_iv_text_embed.csv"
    },

    "hyper_para": {
        "epochs": 200, 
        "lr": 5e-4, 
        "batch_size": 512, 
        "num_train_steps": 1000, 
        "unet_kernel_size": 7, 
        "unet_num_level": 5, 
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

        "text": "Sinus rhythm.", 
        "@text": "text report for generation", 

        "text_embed_setting": {
            "use_api": false, 
            "@use_api": "whether to request OpenAI API", 

            "OPENAI_API_KEY": "",  
            "@OPENAI_API_KEY": "if yes, fill with your API KEY", 

            "loacal_text_embed_path": "./text2ecg/prerequisites/mimic_iv_text_embed.csv", 
            "@local_text_embed_path": "if not, interrogate locally"
        },
        "age": 50, 
        "hr": 90, 
        "gender": "M"
    }
}
```

## Dataset

There are two kinds of dataset which contains pre-encoded latents coupled with text report and related patient specific information. We should prepare either of them ahead of time to speed up the training phase. 

The first kind is a dictionary file whose keys are indices and values are the training samples. We offer the file of dictionary dataset [here](https://huggingface.co/Laiyf/DiffuSETS) associated with the VAE model, You can also create your dictionary datset related to your own VAE model following the instruction [here](#building-your-own-dictdataset). The disk usage of our VAE encoding mimic ecg dataset is about 2 GB, which means it will take up 2 GB after loading into the memory, please check your hardware before running the scripts.

The second kind is spliting the dictionary into single '.pt' items and place them under certain folder, you can use this pattern if the memory limits. See the `VAE_MIMIC_IV_ECG_DATASET` class in `./dataset/mimic_iv_ecg_dataset.py` for implementation details.

## DIY 

### Training your own VAE model 

The VAE model is trained by original mimic iv ecg dataset, please refer to `./vae/vae_train.py` for dependency specification. Use the command below to start training phase, whose result will be saved in `./checkpoints/vae_#` by default.

```sh
python -m vae.vae_train
```

### Building your own DictDataset

1.Update the VAE path and original mimic iv and mimic ecg dataset path in `./utils/vae_encoding.py`, use the following command to initiate script.

```sh
python -m utils.vae_encoding 
```

2.The label of sample in the dictionary attained in the first step contains all of the reports related to. If you want to split every report to a sampple, please concern running
```sh
python -m utils.split_dataset
```
This script leads to the dataset of 2,742,409 entries, which is exactly the form we have adapted to train our UNET in DiffuSETS.