---
title : "경사하강법(Gradient Descent Algorithm)"
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

# 경사하강법(Gradient Descent Algorithm)


## 1. 개념  
어떤 모델에 대한 비용(Cost)를 최소화시키는 알고리즘. ML&DL 모델에서 사용되는 가중치의 최적해를 구할 때 널리 쓰이는 알고리즘임. 기본적인 개념은 함수의 기울기를 계산해 기울기가 낮은 쪽으로 계속 이동시켜서 극값에 이를 때 까지 반복하는 것임. 즉, 비용함수의 반환값=예측값과 실제값의 차이가 작아지는 방향으로 W(기울기)를 지속해서 보정해 나가는 것임.

## 2. 사용하는 이유
비용 함수가 최소가 되는 W를 구할 때에 W 파라미터의 개수가 적다면 고차원 방정식으로 비용 함수가 최소가 되는 W를 도출할 수 있지만, W 파라미터가 많은 경우에는 고차원 방정식을 세우더라도 해결하기가 어려움. 이에 대해 경사하강법은 훨씬 직관적으로 해결이 가능하다는 이점이 있음.
  

## 3. 계산방법
가속도를 속도의 미분값으로 구할 수 있는것처럼 이와 마찬가지로 2차함수의 최저점은 해당 함수의 미분 값인 1차 함수의 기울기가 가장 최소가 될 때임. 예를 들어 비용함수가 아래와 같은 포물선 형태의 2차함수라면 경사하강법은 첫 W에서 미분을 통해 값을 얻은 뒤 미분 값이 계속 감소하는 방향으로 W를 업데이트 하는 것임.

![정의](../../assets/images/post_images/2023-09-05-(01)/figure1.png){: .align-center  width="30%" height="30%"}

비용 함수는 $RSS(w_0, w_1)=1/n*\Sigma^n_{i=1}(y_i-(w_0+w_1*x_i))^2$ 이다. 이를 미분해서 미분 함수의 최솟값을 구하기 위해서는 $w_0, w_1$ 에 대해 편미분을 적용해야 한다. 편미분하면 다음과 같다.

$\frac{\delta R(w)}{\delta w_1}=\frac{2}{N}\Sigma-x_t*(y_i-(w_0+w_1x_i))=-\frac{2}{N}\Sigma x_i*(실제값_i -예측값_i)$

$\frac{\delta R(w)}{\delta w_0}=\frac{2}{N}\Sigma-(y_i-(w_0+w_1x_i))=-\frac{2}{N}\Sigma(실제값_i -예측값_i)$

결과적으로 $w_1, w_0$의 편미분 결과값을 반복적으로 보정하면서 비용 함수$R(w)$가 최소가 되는 $w_1, w_0$ 구함. 업데이트는 새로운 $w_1$을 이전 $w_1$에서 편미분한 결과를 빼면서 적용하는데 위 편미분 값이 너무 클 수 있기 때문에 보정 계수 $\eta$를 곱하고 이를 학습률이라함. 즉 경사하강법은 아래와 같이 계산된다.

새로운 $w_1=이전 w_1+\eta\frac{2}{N}\Sigma x_i*(실제값_i-예측값_i)$  
새로운 $w_0=이전 w_0+\eta\frac{2}{N}\Sigma (실제값_i-예측값_i)$

## 4. 파이썬 코드 구현
```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

np.random.seed(0)
# y = 4X+6을 근사(w1=4, w0=6). 임의의 값은 노이즐 위해 만듦

X = 2*np.random.rand(100,1)
y = 6 + 4 * X + np.random.randn(100,1)

# 산점도 시각화
plt.scatter(X,y)
```

    
![정의](../../assets/images/post_images/2023-09-05-(01)/output_0_1.png){: .align-center  width="55%" height="55%"}
    



```python
# 비용함수 정의
def get_cost(y, y_pred):
    cost = np.sum(np.square(y-y_pred))/N
    return cost
```


```python
# 지속해서 업데이트 되는 w1과 w0를 반환하는 함수
def get_weight_updates(w1, w0, X, y, learning_rate=0.01):
    N = len(y)
    w1_update = np.zeros_like(w1)
    w0_update = np.zeros_like(w0)
    y_pred = np.dot(X, w1.T) + w0
    diff = y - y_pred
    
    w0_factors = np.ones((N,1))
    
    w1_update = -(2/N)*learning_rate*(np.dot(X.T, diff))
    w0_update = -(2/N)*learning_rate*(np.dot(w0_factors.T, diff))
    
    return w1_update, w0_update
```


```python
# 지속해서 업데이트 되는 w1과 w0를 실제 경사하강방식으로 적용하는 함수
def gradient_descent_steps(X, y, iters=10000):
    w0 = np.zeros((1, 1))
    w1 = np.zeros((1, 1))
    
    for ind in range(iters):
        w1_update, w0_update = get_weight_updates(w1, w0, X, y, learning_rate=0.01)
        w1 = w1 - w1_update
        w0 = w0 - w0_update
        
    return w1, w0
```


```python
# 예측 오류 계산
def get_cost(y, y_pred):
    N = len(y)
    cost = np.sum(np.square(y - y_pred))/N
    return cost

w1, w0 = gradient_descent_steps(X, y, iters=1000)
print("w1:{0:.3f} w0:{1:.3f}".format(w1[0,0], w0[0,0]))
y_pred = w1[0, 0]*X+w0
print('Gradient Descent Total Cost:{0:.4f}'.format(get_cost(y, y_pred)))
```

    w1:4.022 w0:6.162
    Gradient Descent Total Cost:0.9935
    


```python
plt.scatter(X,y)
plt.plot(X, y_pred)
```    
![정의](../../assets/images/post_images/2023-09-05-(01)/output_5_1.png){: .align-center  width="55%" height="55%"}
    



## 5. 생각해보기
### Q1. 비용함수가 물결 모양의 2차함수일 때 최저점 계산은 어떻게?
### Q2. w가 n개라면($w_n$) 계산은 어떻게??

## 참고
[경사하강법](http://matrix.skku.ac.kr/sglee/)  
[파이썬 머신러닝 완벽가이드, 권철민, 312p]()



