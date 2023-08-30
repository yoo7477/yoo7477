---
title : "임베딩(Embedding)"
categories :
- DAILY
tag : [MATH, ML]
toc: true
toc_sticky: true
toc_label : "목록"
author_profile : false
search: true
use_math: true
---
<br/>

# 임베딩(Embedding)


## 1. 개념  
고차원 공간에서 텍스트, 이미지와 같은 데이터 조각을 숫자로 표현한 벡터 뭉치. 데이터 처리 시 각 데이터에 일대일로 대응하는 벡터를 만들어 이를 밀집된 벡터 뭉치로 두게 되는데 이를 임베딩이라함.

## 2. 사용하는 이유
- 효율적인 검색 : 대규모 데이터베이스에서 유사하거나 관련 있는 문서를 빠르게 검색하는데 사용할 수 있음. 검색 엔진, 추천 시스템, 문서 클러스터링 등에서 사용할 수 있음.
- 요구되는 스토리지 감소 : 고정된 크기의 숫자 벡터로 저장되기 때문에 텍스트나 이미지로 저장될 때보다 훨씬 적은 용량을 사용함
- 더 빠른 계산 : ML & NLP에서 효율적으로 처리가 가능함.
- 성능 향상 : 임베딩을 통해 빈도나 어순, 단어와 구문 사이의 관계를 파악할 수도 있으므로, 다양한 자연어 처리 작업에서 더 나은 성능을 발휘함.
- 전이학습 : 사전 훈련된 임베딩 값을 통해 새로운 도메인의 모델을 훈련하는데 사용할 수 있고, 이는 시간과 계산 리소스를 절약할 수 있음.
- 구애받지 않는 언어 : 텍스트를 숫자 벡터화 하여 표현하기 때문에 언어에 구애받지 않고 의미론적 유사성을 파악할 수 있음.
- 편리한 운용 : 숫자 벡터의 공통된 표현으로 사용되므로, 여러 소스의 데이터와 쉽게 결합하고 처리할 수 있음. 이는 시스템이나 어플리케이션 간의 운용성을 촉진함.
  

## 3. 변환방법
텍스트를 word 기반의 다수의 피처로 추출하고 이 피처에 단어 빈도수와 같은 숫자 값을 부여하면 텍스트는 단어의 조합인 벡터값으로 표현될 수 있음. 단어의 빈도수 기반 변환이 가장 기본적인 임베딩이라고 볼 수 있음. 이를 피처 벡터화(Feature Vectorization) 또는 피처 추출(Feature Extraction)이라고 함. 대표적인 방법에는 BOW(Bag of Words)와 Word2Vec 방법이 있음. 

## 4. 임베딩 종류
a. 행렬 분해 기반: 말뭉치(Corpus) 정보가 들어 있는 원래 행렬을 두 개 이상의 작응 행렬로 쪼개는(Decomposition) 방식의 임베딩, ex) GloVe, Swivel
b. 어떤 단어 주변에 특정 단어가 나타날지 예측하고, 이전 단어들이 주어졌을 때 다음 단어가 무엇일지 예측하거나, 문장 내 일부 단어를 지우고 해당 단어가 무엇일지 맞추는 과정에서 학습하는 방식. ex) Word2Vec, FastText, BERT, ELMo, GPT 등
c. 주어진 문서에 잠재된 주제를 추론하는 방식 ex) LDA

## 5. 생각해보기
### Q1. text-embedding-ada-002이란?
OpenAI가 개발한 모델로 23년 8월 30일 기준 Hugging Face에 MTEB 리더보드 13위에 순위한 모델임. 특정 도메인을 위한 모델 보다는 범용 임베딩을 위한 모델임. 여러 언어를 지원하고, 미세 조정 기능이 있음. 8192자(byte)까지 지원가능하며 다른 임베딩 모델에 비해 압도적임.
## 참고
[임베딩-wiki](https://namu.wiki/w/%EC%9E%84%EB%B2%A0%EB%94%A9)  
[Text Embedding-What, Why, and How?](https://medium.com/@yu-joshua/text-embedding-what-why-and-how-13227e983ba7)  
[OpenAI 임베딩 모델 비교](https://www.educative.io/answers/text-embedding-ada-002-vs-openais-older-embedding-models)  
[MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
[한국어 임베딩, 이기창]()
