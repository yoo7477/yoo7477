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





### 3.1 절차
---
1) 고차원의 원 공간에서 데이터 간 유사도$p_{ij}$를 정의함.
2) 저차원의 임베딩(축소) 공간에서 데이터 간 유사도$q_{ij}$를 정의함.
3) 저차원의 공간이 고차원 공간에 가까워지도록 gradient descent로 계산해 데이터를 변환함.
---
1) 고차원의 원 공간에서 데이터 간 유사도$p_{ij}$를 정의함.
   
   $p_{ij}$를 정의하기 위해 점 $x_i$에서 $x_j$로의 유사도인 $p_{j|i}$를 정의해야함. 그래서 먼저 기준점 $x_i$에서 다른 모든 점들과의 유클리드 거리 $|x_i-x_j|$를 계산함. 그리고 이 거리를 기반으로 기준 점과 다른 점들 간의 거리가 얼마나 가까운지를 확률로 나타냄. 확률로 나타내는 방법은 점과 점의 거리를 $\sigma_i$로 나누고 negative expoential을 취하면 $\exp(-|x_i-x_j|^2/2\sigma_i^2)$이 됨. 그리고 모든 점들과의 거리의 합은 $\exp(-|x_i-x_k|^2/2\sigma_i^2)$으로 각각을 나눠주면 아래와 같은 확률 형식이 됨. 

   $p_{j|i}$=$\frac{\exp(-|x_i-x_j|^2/2\sigma_i^2)}{\sum_{k\ne i}\exp(-|x_i-x_k|^2/2\sigma_i^2)}$

   여기서 $\sigma_i$는 모든 점마다 다르게 정의 되는데 t-SNE가 안정적인 학습 결과를 가지게 되는 부분임. 결과적으로 점과 점사이의 거리가 멀어서 분모가 작아지게 되면 $p_{j|i}$도 작아지게 되고 점과 점사이가 가까우게 되면 유사도가 커져서 유사도가 커지는 방향으로 점들이 이동하는 모델임. 점 간의 유사도를 대칭적으로 만들기 위하여 $p_{j|i}$와 $p_{i|j}$의 평균으로 두 점간의 유사도를 정의함. 그리고 $n$개의 점으로 나눠주면 모든 점들간의 유사도의 합이 1이 되도록 만들 수 있고 확률의 개념을 사용할 수 있는 것임. 따라서 원 공간에서의 유사도 $p_{ij} = \frac{p_{i|j}+p_{j|i}}{2n}$으로 정의가능함.

2) 저차원의 임베딩(축소) 공간에서 데이터 간 유사도$q_{ij}$를 정의함.

    저차원 공간에서의 유사도 $q_{ij}$도 고차원 공간에서의 유사도 $p_{ij}$와 동일하게 확률로 나타낼 수 있도록 $\sum_{ij}q_{ij}=1$로 정의함. 동일하게 두 점간의 거리가 작을수록 유사도는 큰 값을 갖도록 두 점간의 거리에 역수를 취하여 유사도로 이용함. 추가적으로 안정적인 역수를 얻기 위해 거리 값에 1을 더하고 역수를 취해주면 $(1+|y_i-y_j|^2)^{-1}$이 되고 이 값은 확률 분포인 t-분포가 됨. 또한 이도 모든 점들간의 합으로 나눠줌으로써 아래와 같이 유사도를 정의함

    $q_{ij}=\frac{(1+|y_i-y_j|^2)^{-1}}{\sum_{k\ne l}(1+|y_k-y_l|^2)^{-1}}$

3) 저차원의 공간이 고차원 공간에 가까워지도록 gradient descent로 계산해 데이터를 변환함.

    t_SNE는 $p_{ij}$에 가장 가깝도록 $q_{ij}$를 학습함. 이는 정확히 $q_{ij}$를 정의하는 $y_i, y_j$를 학습하는 것이고, 결국에는 정답지인 원 공간의 $p_{ij}$를 보면서 $y_i, y_j$를 이동하는 것과 같음. 학습에는 경사하강법을 이용하며, $q_{ij}$와 $p_{ij}$의 비용함수가 최소가 되는 방향으로 $y$의 점들을 이동함. 이동량은 아래와 같음

    $\frac{\delta C}{\delta y_i}=\sum_j(p_{ij}-q_{ij})(y_i-y_j)\frac{1}{1+|y_i-y_j|^2}$


## 4. 파이썬 코드 구현
Iris data set으로 비교
```python
# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Load the iris dataset
iris = load_iris()

# Convert data to dataframe
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
```
Original data set
```python
# Visualize the original data
plt.scatter(iris_df['sepal length (cm)'], iris_df['sepal width (cm)'], c=iris.target)
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.title('Iris Dataset')
plt.show()
```
![정의](../../assets/images/post_images/2023-09-25-(01)/Figure_1.png){: .align-center  width="30%" height="30%"}

Apply PCA
```python
# Apply PCA
pca = PCA(n_components=2)
iris_pca = pca.fit_transform(iris.data)

# Visualize PCA output
plt.scatter(iris_pca[:,0], iris_pca[:,1], c=iris.target)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title('PCA Output')
plt.show()
```
![정의](../../assets/images/post_images/2023-09-25-(01)/Figure_2.png){: .align-center  width="30%" height="30%"}
Apply t-SNE

```python
# Apply t-SNE
tsne = TSNE(n_components=2)
iris_tsne = tsne.fit_transform(iris.data)

# Visualize t-SNE output
plt.scatter(iris_tsne[:,0], iris_tsne[:,1], c=iris.target)
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.title('t-SNE Output')
plt.show()
```
![정의](../../assets/images/post_images/2023-09-25-(01)/Figure_3.png){: .align-center  width="30%" height="30%"}

## 4. PCA와 t-SNE 비교
### t-SNE
- t-SNE는 local structure와 점 들간 관계는 보존하면서 저차원 공간(2차 평면)에 고차원 데이터를 시각화 하는데 특출남. 그래서 데이터분석에서 데이터 시각화나 데이터 탐색에 주로 사용됨.
- 마찬가지로 고차원에서 식별하기 어려운 군집이나 패턴을 찾는데 유용함.
- 복잡한 데이터와 비선형 구조의 데이터에서 잘 작동함.
### PCA
- PCA는 데이터의 variability를 잘 보존하면서 거대 데이터 세트의 차원을 축소하기 위해 대중적이고 넓게 사용됨.
- PCA도 패턴이나 데이터에서 변수 간의 관계를 찾는데 유용함.
- t-SNE와 비교하여 상대적으로 빠르고 컴퓨터 비용도 효율적임. 
## 5. 생각해보기
### Q1. PCA와 비교하여 t-SNE의 계산비용이 많이 드는 이유는?
- t-SNE는 고차원의 데이터 간 유사도와 저차원의 데이터 간 유사도를 계산하기 위해서는 $n^2$번의 거리 계산을 수행해야됨. 또한 이상치의 영향력을 줄이기 위해 모든 점마다 $\sigma_i$를 다르게 정의하는 것도 작지만 한 몫함. 
### Q2. t-SNE의 많은 계산 비용에 대한 대안은?
- Barnes-hut tree라는 vector indexing 방법을 사용함. 구체적인 방법은 원 공간을 여러 개의 집단으로 나누고 t-SNE처럼 점과 점간의 거리가 아닌 집단과 집단의 거리를 계산하여 점이 아닌 집단을 움직임. 사이킷런의 t-SNE는 이 방식이 적용돼있음.
### Q3. Perplexity란?
- t-SNE는 임베딩 과정에서 nearest 점들과 그렇지 않은 점들을 칼처럼 양분하진 않음. 대신 유사도$(p_{j|i})$ 계산해 거리에 반비례하여 영향력을 정의함. 그리고 perplexity는 어느 범위까지 영향력을 미치게할지 정의함.
- Perplexity는 $2^{entropy}$로 정의됨. 통계학에서 entropy는 확률분포의 불확실성을 나타내는 척도임. 이 불확실성의 척도는 확률분포에 대한 정보량의 기대값으로 표현됨. 즉, 어떠한 사건이 일어날 확률에 대한 불확실성이 클 수록 entropy가 크다고 볼 수 있음. 그리고 $\sigma$를 어떤 값으로 설정하느냐에 따라 $p_{j|i}$의 perplexity가 결정됨.  
$entropy(p_x)=\sum -p_{xi}log(p_{xi})$  
대체로 Perplexity가 매우 작으면 locality가 많이 반영되어 왜곡된 공간이 학습되고 거리 정보가 많이 보존되지만 집단을 구분하기 어렵고, 조금씩 높아질 수록 거리정도도 어느 정도 보존되면서 집단을 구분할 수도 있음. 너무 값이 커지면 점들 사이에 거리 정보가 보존되지 않음. 
[P.S. 이전에 LDA 공부할 때도 Perplexity가 있었는데 따로 포스팅하자]()
### Q4. 모든 점들에 대해 거리를 계산할 때 사용되는 수식은?
- 기본적으로 앞서 절차1)에서 말했던 것처럼 유클리드 거리르를 사용한다. 다만 bag-of-words 모델과 같이 Sparse vector로 표현된 데이터라면 Cosine이나 Jaccard가 더 적절할 수 있다.
## 참고
[t-SNE origin, 2008](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf)  
[t-SNE 관련 web blog1](https://lovit.github.io/nlp/representation/2018/09/28/tsne/)  
[t-SNE 관련 web blog2](https://3months.tistory.com/571)  
[t-SNE 관련 web blog3](https://aaweg-i.medium.com/pca-vs-t-sne-dimensionality-reduction-techniques-fdd7908973a4)


