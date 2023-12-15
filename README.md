# Music to Mutilabel for Image generating Sentence

인공지능(AI)을 활용한 음악 분석은 현대 음악 산업과 예술 분야에 혁신적인 변화를 가져오고 있다. AI는 음악의 특징, 구조, 리듬, 음성 톤 등을 학습하여 음악을 이해하고, 감정 분석과 사용자 맞춤형 음악 추천에 활용된다. 음악 분석을 기반으로 AI는 음악 생성, 합성, 악보 제작 등 창의적인 지원을 할 수 있다. 음악과 시각 예술 간의 상호작용을 위해 AI는 음악을 시각적으로 표현하며, 이는 콘서트, 음악 비디오, 앨범 표지 등에 활용될 수 있다. 

이 과제는 음악 클립에 대한 멀티 태그를 수정, 재정의하고, 이를 기반으로 문장 생성 및 음악을 시각적으로 표현하는 이미지를 생성하는 것을 목표로 한다.


<div align="center">
  <img width="100%" alt="CLMR model" src="https://github.com/dlwns97/CVIP/raw/master/architecture.png">
</div>
<div align="center">
  < 이번 과제의 전체 아키텍처>
</div>

#
This repository relies on CLMR implementation, which can be found [here](https://github.com/Spijkervet/CLMR).
#


## Quickstart
```
git clone https://github.com/spijkervet/clmr.git && cd clmr

pip3 install -r requirements.txt
# or
python3 setup.py install
```

The following command downloads MagnaTagATune, preprocesses it and starts self-supervised pre-training on 1 GPU (with 8 simultaneous CPU workers) and linear evaluation:
```
python3 preprocess.py --dataset magnatagatune

# add --workers 8 to increase the number of parallel CPU threads to speed up online data augmentations + training.
python3 main.py --dataset magnatagatune --gpus 1 --workers 8

python3 linear_evaluation.py --gpus 1 --workers 8 --checkpoint_path [path to checkpoint.pt, usually in ./runs]
```

## Pre-train on your own folder of audio files
Simply run the following command to pre-train the CLMR model on a folder containing .wav files (or .mp3 files when editing `src_ext_audio=".mp3"` in `clmr/datasets/audio.py`). You may need to convert your audio files to the correct sample rate first, before giving it to the encoder (which accepts `22,050Hz` per default).

```
python preprocess.py --dataset audio --dataset_dir ./directory_containing_audio_files

python main.py --dataset audio --dataset_dir ./directory_containing_audio_files
```


## Results

### MagnaTagATune

| Encoder / Model | Batch-size / epochs | Fine-tune head |  ROC-AUC |  PR-AUC |
|-------------|-------------|-------------|-------------|-------------|
| SampleCNN / CLMR | 48 / 10000 | Linear Classifier | 88.7 | **35.6** |
SampleCNN / CLMR | 48 / 10000 | MLP (1 extra hidden layer) |  **89.3** | **36.0** |
| [SampleCNN (fully supervised)](https://www.mdpi.com/2076-3417/8/1/150) | 48 / - | - | 88.6 | 34.4 |
| [Pons et al. (fully supervised)](https://arxiv.org/pdf/1711.02520.pdf) | 48 / - | - | 89.1 | 34.92 |

### Million Song Dataset

| Encoder / Model | Batch-size / epochs | Fine-tune head |  ROC-AUC |  PR-AUC |
|-------------|-------------|-------------|-------------|-------------|
| SampleCNN / CLMR | 48 / 1000 | Linear Classifier | 85.7 | 25.0 |
| [SampleCNN (fully supervised)](https://www.mdpi.com/2076-3417/8/1/150) | 48 / - | - | **88.4** | - |
| [Pons et al. (fully supervised)](https://arxiv.org/pdf/1711.02520.pdf) | 48 / - | - | 87.4 | **28.5** |


## Pre-trained models
*Links go to download*

| Encoder (batch-size, epochs) | Fine-tune head | Pre-train dataset | ROC-AUC | PR-AUC
|-------------|-------------|-------------|-------------|-------------|
[SampleCNN (96, 10000)](https://github.com/Spijkervet/CLMR/releases/download/2.0/clmr_checkpoint_10000.zip) | [Linear Classifier](https://github.com/Spijkervet/CLMR/releases/download/2.0/finetuner_checkpoint_200.zip) | MagnaTagATune |  88.7 (89.3) | 35.6 (36.0)
[SampleCNN (48, 1550)](https://github.com/Spijkervet/CLMR/releases/download/1.0/clmr_checkpoint_1550.pt) | [Linear Classifier](https://github.com/Spijkervet/CLMR/releases/download/1.0-l/finetuner_checkpoint_20.pt) | MagnaTagATune | 87.71 (88.47) | 34.27 (34.96)

## Training
### 1. Pre-training
Simply run the following command to pre-train the CLMR model on the MagnaTagATune dataset.
```
python main.py --dataset magnatagatune
```

### 2. Linear evaluation
To test a trained model, make sure to set the `checkpoint_path` variable in the `config/config.yaml`, or specify it as an argument:
```
python linear_evaluation.py --checkpoint_path ./clmr_checkpoint_10000.pt
```

## Configuration
The configuration of training can be found in: `config/config.yaml`. I personally prefer to use files instead of long strings of arguments when configuring a run. Every entry in the config file can be overrided with the corresponding flag (e.g. `--max_epochs 500` if you would like to train with 500 epochs).

## Logging and TensorBoard
To view results in TensorBoard, run:
```
tensorboard --logdir ./runs
```
