---
title : "혼잡도(Perplexity)"
categories :
- DAILY
tag : [ML]
toc: true
toc_sticky: true
toc_label : "목록"
author_profile : false
search: true
use_math: true
---
<br/>

# 혼잡도(Perplexity)


## 1. 개념  
Perplexity는 사전적 의미로 당혹스러운[이해하기 힘든] 것을 말함. 기존에는 LDA나 t-SNE 등에서 확률 모델을 평가할 때 주로 사용하는 지표였음. 다만 최근 ChatGPT 히트 이후에는 LLM(Large Language Model)에 대한 관심이 커지면서 가장 일반적인 언어 모델 평가지표 중 하나로 사용되고 있음. 

## 2. 사용하는 이유
LDA 등의 토픽 모델링에서는 클러스터링 기법을 평가하는 의미로 쓰였었고, 현재는 언어모델을 평가하는 척도로써 사용됨. 평가 방법에는 외재적(Extrinsic) 평가와 내재적(Intrinsic) 평가로 나뉨. 각각의 의미는 아래와 같음.
- 외재적평가 : 언어모델을 특정 task에 적용해서 Loss 나 Accuracy를 사용하여 확인하는 방법임. 즉 해당 언어모델이 어디어디에 쓰일 수 있는 지를 바탕으로 외부적 관점에서 평가하는 것을 말함.
- 내재적평가 : 언어모델을 특정 task에 적용하지 않고, 자체적인 성능을 평가하는 것을 말함. 즉, 기법 결과의 자체가 실제로 잘 분류하는지, 분류된 결과가 사람이 판단하기에 적합한지 등을 내부적 관점에서 평가하는 것을 말함. Perplexity는 내부적 평가에 해당하며, 토픽모델링에서는 Perplexity의 한계를 극복하기 위해 Topic Coherenece의 척도도 함께 쓰길 권장함. 다만 언어모델에서는 단어 시퀀스의 확률을 계산하는 것이 중요하기 때문에 유사 단어를 활용한 점수매기기(?)의 Coherence는 사용하지 않음.
  
## 3. 계산방법
Perplexity는 test set에 대한 정규화된 역확률로 표현할 수 있으며, 이는 아래와 같음.  
$PP(W) = \sqrt[N]{\frac{1}{P(w_1,w_2,...,w_N)}}$  
여기서 ${P(w_1,w_2,...,w_N)}$를 설명하면 1부터 N까지의 단어 시퀀스로 구성된 문장에서 다음에 올 단어를 예측할 확률을 의미함. 예를들어, P(pizza|For Dinner I'm make __) > P(tree|For Dinner I'm make __) 처럼 For dinner I'm make의 다음 단어가 tree 보다는 pizza가 나올 확률이 높은 것처럼 언어모델에서는 올바른 문장에 더 높은 확률을 계산하는 모델을 필요로 하는데 이를 평가하는 지표가 Perpelxity라고 볼 수 있는 것임. 수식에서도 볼 수 있듯이 Perplexity가 낮은 모델이 더 높은 성능을 보이는 것임.

### 3.1 정규화
다만 데이터 세트에 단어가 너무 많다면 단어의 수가 적은 데이터 세트보다 당연히 확률이 낮게 나올 수 밖에 없음. 그래서 총 단어 수로 test set의 확률을 정규화하여 data set의 크기와 무관한 값을 얻는 것이 필요함. 정규화하는 과정은 아래와 같음
- (1) $P(W) = P(w_1, w_2,...,w_N)=P(w_1)P(w_2)...P(w_N)=\prod^N_{i=1}P(w_i)$
- (2) 곱의 형태를 합의 형태로 변환하기 위한 log화 $ln(P(W))=ln(\prod^N_{i=1}P(w_i))=\sum^N_{i=1}lnP(w_i)$
- (3) 총 단어수로 나누고 단어별 로그 확률 계산 $\frac{\sum^N_{i=1}lnP(w_i)}{N}$
- (4) 지수화를 활용 로그 제거 $e^{\frac{\sum^N_{i=1}lnP(w_i)}{N}}$
- (5) -N승하여 정규화 $P(W)^{1/N}=(\prod^N_{i=1}P(w_i))^{1/N}$
- (6) $PP(W) = \frac{1}{P(w_1,w_2,...,w_N)^{1/N}}=\sqrt[N]{\frac{1}{P(w_1,w_2,...,w_N)}}$
- (결론) Perplexity는 test set의 단어 수(N)으로 정규화된 test set의 역확률로 해석할 수 있음.

### 3.2 Perpelxity와 Cross-Entropy
Perplexity는 Cross Entropy의 지수로 정의될 수도 있음. Entropy란 t-SNE 게시물에서도 언급했던 것처럼 통계학에서 확률분포의 불확실성을 나타내는 척도임. 이전 정의와 동일한지는 아래와 같이 확인가능함.
$PP(w) = 2^{H(W)} = 2^{-\frac{1}{N}log_2P(w_1,w_2,...,w_N)} = (2^{log_2P(w_1,w_2,..,w_N)})^{1/N}=\sqrt[N]{\frac{1}{P(w_1,w_2,..,w_N)}}$
따라서 cross-entropy는 $E(p)=H(p,q)-\sum_ip(x_i)log_2q(x_i)$와 같이 정의된다고 볼 수 있고 여기서 p는 언어의 실제 확률 분포이나 알 수 없는 값이고 q는 test set이 아닌 training set으로 부터 추정된 확률 분포로 계산할 수 있는 것임. 그래서 단어 W의 시퀀스가 충분히 길거나 N이나 충분히 크면 우리는 각각의 단어를 Shannon-McMilan-Breiman theorem에 따라 cross-entropy를 아래와 같이 근사시킬 수 있는 것임.  
$H(p,q)\approx-\frac{1}{N}log_2q(W)$   
이를 다시 지수화하여 Perplexity를 cross-entropy를 활용해 정의할 수 있는 것이고, 이는 test set에 대한 loss를 계산하는 것과 동일한 것임. 
(여러 게시물에는 cross-entropy의 log가 상용로그도 쓰고 자연로그도 쓰는데 다 틀렸고 밑이 2인 log를 써야하니 명심할 것......)

## 5. 생각해보기
아이고 생각이고 뭐고 생각보다 간단한 내용인데 하기가 싫었는지 글을 읽어도 이해도 안되고 진도도 너무 안나가서 고생이 많았다. 추가로 궁금한 내용이 있거나 다른 공부하면서 추가적으로 생기면 추가하도록 하자......
## 참고
[Perplexity 관련 best post](https://towardsdatascience.com/perplexity-in-lan%20tguage-models-87a196019a94)  
