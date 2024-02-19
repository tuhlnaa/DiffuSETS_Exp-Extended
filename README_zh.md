# DiffuSETS

本仓库所有python指令均运行在仓库根目录下（`/.../text2ecg/`）

对于根目录下的python脚本，使用 `python xxx.py` 运行，对于子文件夹下的脚本，使用 `python -m aaa/.../xxx` 运行。如此，所有脚本运行时路径均为仓库根目录。

## 生成与评估

生成
```sh
python DiffuSETS_inference.py
```

评估详见 `metrics.ipynb`

## 训练

初始训练脚本为 `DiffuSETS_train.py`，其中使用到了所有的condition和诊断文本embedding，启动方式为
```sh
python DiffuSETS_train.py
```

如果要进行消融实验，建议自行创建对应的文件夹，然后使用如下方式启动
```sh
python -m aaa/.../xxx
```

## 仓库结构

### checkpoints

存放各种实验的记录和保存的模型

### dataset

pytorch数据集读取模块，包含mimic iv ecg和ptbxl以及相应的latent读取

### unet

扩散模型中使用到的Unet

### vae

vae模型，训练，评估和数据集编码脚本

### clip

clip模型，训练和测试脚本