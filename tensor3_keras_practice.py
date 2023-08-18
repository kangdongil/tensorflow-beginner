'''
IS_ADMIT / GRE / GPA / UNIV_RANK
427개의 영미권대학 지원학생의 합격여부에 관한 데이터가 주어진다
GRE가 760점, GPA가 3.0, 지원하는 대학교랭킹이 2일 경우 이 학생은 합격할지 여부를 확률로 나타내보시오.
'''

import numpy as np
import pandas as pd
import tensorflow as tf

# CSV파일 불러오고 전처리하기
data = pd.read_csv('gpascore.csv')
# print(data.isnull().sum()) # 공백오류 점검하기
data = data.dropna() # 공백 데이터 제거하기
# data = fillna(100) # 공백에 default값 넣기
# print(data['gpa'].min()) # 특정 열의 최솟값 구하기
# print(data['gre'].count()) # 열의 데이터 개수 구하기
train_x = []
train_y = data['admit'].values # 1차원 리스트 만들기

for i, rows in data.iterrows(): # 한 행씩 출력하기
    x_record = [rows['gre'], rows['gpa'], rows['rank']]
    train_x.append(x_record)

# DeepLearning Model 설계하기
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='tanh'), # 레이어 / 노드수
    tf.keras.layers.Dense(128, activation='tanh'),
    tf.keras.layers.Dense(1, activation='sigmoid'), # 결과는 노드 한개. 결과값을 0~1사이로 압축해주는 sigmoid 사용.
]) # 모델 / 원하는 결과물를 도출하도록 모델을 설계하기

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) # binary_crossentropy는 확률문제에 적합한 손실함수
model.fit(np.array(train_x), np.array(train_y), epochs=1000) # numpy array로 변환

# 문제 예측
forcast_y = model.predict([[760, 3.0, 2], [400, 2.2, 1]])
print(forcast_y)