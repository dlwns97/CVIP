# Music to Mutilabel for Image generating Sentence
## Sogang Univ. Computer Vision and Image Processing lab.
## Big-data computing project

인공지능(AI)을 활용한 음악 분석은 현대 음악 산업과 예술 분야에 혁신적인 변화를 가져오고 있다. AI는 음악의 특징, 구조, 리듬, 음성 톤 등을 학습하여 음악을 이해하고, 감정 분석과 사용자 맞춤형 음악 추천에 활용된다. 음악 분석을 기반으로 AI는 음악 생성, 합성, 악보 제작 등 창의적인 지원을 할 수 있다. 음악과 시각 예술 간의 상호작용을 위해 AI는 음악을 시각적으로 표현하며, 이는 콘서트, 음악 비디오, 앨범 표지 등에 활용될 수 있다. 

이 과제는 음악 클립에 대한 멀티 태그를 수정, 재정의하고, 이를 기반으로 문장 생성 및 음악을 시각적으로 표현하는 이미지를 생성하는 것을 목표로 한다.


### Ref
This repository relies on CLMR implementation, which can be found [here](https://github.com/Spijkervet/CLMR).


## 과제 목표
 25,863개의 음악 클립으로 구성된 MagnaTagATune 데이터셋을 사용하여, Contrastive Learning of Musical Representations (CLMR) 모델로 음악의 태그를 추출하고 이미지 생성하는 것이 주된 목표이다. 데이터셋의 태그를 재정의하여 음악 클립과 태그 간의 정확성을 향상시키는 것이 중요한 과제이다. 태그를 기반으로 LLM을 사용하여 문장을 생성하고, 이를 이미지 생성 모델의 입력으로 사용한다. 생성된 이미지는 음악 클립의 특성을 시각적으로 표현하는 데 사용되며, 인간 중심 평가를 통해 비교 분석될 예정이다. 이 과제는 음악과 시각 예술의 통합을 탐구하고, AI를 통한 창의적인 예술 표현의 새로운 가능성을 제시한다.
 
## 과제 수행 방법
 이 프로젝트에서는 MagnaTagATune 데이터셋을 사용하여 음악 클립의 태그를 재정의하고, 이를 기반으로 음악 정보 검색 연구를 진행한다. 데이터셋에는 다양한 장르의 음악 클립과 태그가 포함되어 있다. 데이터셋의 중요한 문제점으로 중복되는 의미의 태그가 많다는 것이 확인되었다. 중복 태그를 통일하여 데이터셋의 명확성을 개선한다. Voice Activity Detection(VAD) 기술을 사용하여 잘못된 태그를 수정한다. 이를 위해 Silero-VAD 알고리즘을 활용하여 목소리 관련 태그의 정확성을 향상시킨다. CLMR 모델을 사용하여 음악 클립의 태그를 예측하고, 이를 바탕으로 fine-tuning을 통해 모델의 성능을 개선한다. ChatGPT를 사용하여 태그를 기반으로 문장을 생성하고, 이 문장을 Stable Diffusion 모델의 입력으로 사용하여 이미지를 생성한다. Stable Diffusion을 통해 이미지를 생성하고, 재정의된 태그와 기존 태그로부터 생성된 이미지를 비교하여 이미지 생성 결과를 human evaluation 방법을 이용하여 평가한다.

<div align="center">
  <img width="100%" alt="CLMR model" src="https://github.com/dlwns97/CVIP/raw/master/architecture.png">
</div>
<div align="center">
  < 이번 과제의 전체 아키텍처 >
</div>


## Results

### 생성된 이미지
<div align="center">
  <img width="80%" alt="Music" src="https://github.com/dlwns97/CVIP/blob/master/%EC%9D%8C%EC%95%85%20%EC%83%9D%EC%84%B1%20%EC%9D%B4%EB%AF%B8%EC%A7%80.png">
</div>
<div align="center">
  < 이미지 생성 결과 >
</div>

### 평가 방식
<div align="center">
  <img width="80%" alt="googleform" src="https://github.com/dlwns97/CVIP/blob/master/form.png">
</div>
<div align="center">
  < 평가 구글 폼 양식 >
</div>

### 평가 결과
 평가 방법 설명에 따라, 음악과 그 음악에 해당하는 1~4번까지의 4개의 이미지를 제시하였다. 응답자는 먼저 사운드 클라우드에 업로드한 음악을 듣고 가장 선호하는 이미지와 두 번째로 선호하는 이미지를 선정하는 방식으로 설문을 진행하였다. 응답자들은 1~4번 이미지가 어떤 방식으로 이미지들이 생성되는지 알 수 없고, 오직 음악과 이미지만을 보고 설문을 진행한다. 총 30명에게 그림 24과 같은 6개의 항목에 대해 설문을 진행하였다. 
 설문 결과는 표1과 같이, 모든 항목에 대해서 2번이 총 341점으로 가장 많은 점수를 획득하였고, 1번이 247점으로 다음으로 높은 점수를 획득하였다. 각 항목별 결과에 따르면 태그 재정의가 이루어진 이후 태그들로 문장을 입력하였을 때 생성된 이미지인 두 번째 이미지가 6개의 항목 중 3개의 항목에서 가장 높은 점수를 획득하고 1개의 항목에서 첫 번째 이미지와 동점을 기록하였다. 또한, 전반적으로 태그 재정의가 이루어지기 이전의 태그들을 입력으로 한 세 번째 이미지와 태그 재정의가 이루어지기 이전의 태그들로 생성된 문장을 입력으로 한 네 번째 이미지는 아주 낮은 투표수를 보이거나 아무 선택도 받지 못하였다. 
 문장을 입력으로 한 이미지들과 태그(단어)들을 입력으로 한 이미지의 차이를 보면 문장을 입력으로 한 두 번째와 네 번째 이미지보다 태그들을 입력으로 한 첫 번째와 세 번째 이미지가 94점이 낮은 것을 볼 수 있다. 
 이를 통해 태그 재정의를 통해 음악에 알맞은 태그들로 수정되었고 그것들을 이용하여 이미지를 생성할 때, 단어들을 입력하는 것보다는 문장을 생성하여 입력하는 것이 많은 사람들이 선호한다는 것을 알 수 있었다. 

 <div align="center">
  <img width="80%" alt="result" src="https://github.com/dlwns97/CVIP/blob/master/result.png">
</div>
<div align="center">
  < 평가 결과 표 >
</div>

