# Music to Mutilabel for Image generating Sentence
# Sogang Univ. Computer vision and image processing lab.
# Big-data computing project

인공지능(AI)을 활용한 음악 분석은 현대 음악 산업과 예술 분야에 혁신적인 변화를 가져오고 있다. AI는 음악의 특징, 구조, 리듬, 음성 톤 등을 학습하여 음악을 이해하고, 감정 분석과 사용자 맞춤형 음악 추천에 활용된다. 음악 분석을 기반으로 AI는 음악 생성, 합성, 악보 제작 등 창의적인 지원을 할 수 있다. 음악과 시각 예술 간의 상호작용을 위해 AI는 음악을 시각적으로 표현하며, 이는 콘서트, 음악 비디오, 앨범 표지 등에 활용될 수 있다. 

이 과제는 음악 클립에 대한 멀티 태그를 수정, 재정의하고, 이를 기반으로 문장 생성 및 음악을 시각적으로 표현하는 이미지를 생성하는 것을 목표로 한다.


<div align="center">
  <img width="100%" alt="CLMR model" src="https://github.com/dlwns97/CVIP/raw/master/architecture.png">
</div>
<div align="center">
  < 이번 과제의 전체 아키텍처>
</div>


### Ref
This repository relies on CLMR implementation, which can be found [here](https://github.com/Spijkervet/CLMR).


## Pre-train on your own folder of audio files
Simply run the following command to pre-train the CLMR model on a folder containing .wav files (or .mp3 files when editing `src_ext_audio=".mp3"` in `clmr/datasets/audio.py`). You may need to convert your audio files to the correct sample rate first, before giving it to the encoder (which accepts `22,050Hz` per default).

```
python preprocess.py --dataset audio --dataset_dir ./directory_containing_audio_files

python main.py --dataset audio --dataset_dir ./directory_containing_audio_files
```


## Results

<div align="center">
  <img width="100%" alt="Music6" src="https://github.com/dlwns97/CVIP/blob/master/music_6.png">
</div>
<div align="center">
  < 음악 6의 생성 결과 >
</div>




