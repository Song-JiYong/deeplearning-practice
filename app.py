# <tensorflow를 이용한 대학 합격 예측 AI 만들기>

import pandas as pd

data = pd.read_csv('gpascore.csv')   # csv 파일 읽어오기

# 데이터 전처리
#print(data.isnull().sum())
data = data.dropna() # dropna(): NaN/빈값 행 제거
#print(data.isnull().sum())
#data.fillna(100) # fillna(N): 빈칸을 N값으로 채워줌
# print(data['gpa'].min()) # data[]: 열 출력 / .min() .count()

# y입력값 만들기(label) y = [정답1, 정답2, 정답3...]
y데이터 = data['admit'].values

# x입력값 만들기 x = [[데이터1],[데이터2],[데이터3]]
x데이터 = [ ]
for i, rows in data.iterrows():
    x데이터.append([ rows['gre'], rows['gpa'], rows['rank'] ])


import numpy as np  # 일반 리스트 -> numpy array 변환(fit에 집어넣기 위함), 다차원 리스트, 행렬 만들기
import tensorflow as tf

# 1. 딥러닝 model 디자인하기
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation='tanh'),   # 히든 레이어1, (괄호 노드 개수, 활성함수)
    tf.keras.layers.Dense(64, activation='tanh'),   # 히든 레이어2
    tf.keras.layers.Dense(1, activation='sigmoid'), # 히든 레이어3, 출력노드는 1, sigmoid: 0~1값 출력
])

# 2. model 컴파일하기
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) # loss함수 binary~: 0~1 결과(분류/확률)

# 3. model 학습(fit) 시키기
model.fit( np.array(x데이터), np.array(y데이터), epochs=1000 )   # (학습데이터, 실제정답, 학습횟수)


# 결과창, loss: 현재 총손실, accurancy: 모델의 정답률.
# epochs 수를 높이면 비교적 낮은 loss와 높은 accurancy를 얻을 수 있음
# 모델 성능 향상 - 데이터전처리, 모델 튜닝

# 4. 예측
예측값 = model.predict( [ [750, 3.70, 3], [400, 2.2, 1] ] ) # 1에 가까울수록 합격률 높음
print(예측값)