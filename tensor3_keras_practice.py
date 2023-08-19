'''
IS_ADMIT / GRE / GPA / UNIV_RANK
427개의 영미권대학 지원학생의 합격여부에 관한 데이터가 주어진다
GRE가 760점, GPA가 3.0, 지원하는 대학교랭킹이 2일 경우 이 학생은 합격할지 여부를 확률로 나타내보시오.
'''

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import models, layers

# CSV파일 불러오고 전처리하기
df = pd.read_csv('download/gpascore.csv')
# print(df) # 데이터프레임 시각화
# print(df.isnull().sum()) # 공백오류 점검하기
df = df.dropna() # 공백 데이터 제거하기
# df = fillna(100) # 공백에 default값 넣기
# print(df['gpa'].min()) # 특정 열의 최솟값 구하기
# print(df['gre'].count()) # 열의 데이터 개수 구하기

# df[['gre', 'gpa', 'rank']] # 여러 특정 행만 추출
# df.iloc[:, 1:4] # 인덱스 값범위로 추출
# df.loc[:, 'gre':'rank'] # 행이름 범위로 추출
train_x = df.loc[:, 'gre':'rank'].values
train_y = df['admit'].values # 1차원 리스트 만들기

'''
train_x = []
for i, rows in df.iterrows(): # 한 행씩 출력하기
    x_record = [rows['gre'], rows['gpa'], rows['rank']]
    train_x.append(x_record)
'''

# DeepLearning Model 설계하기
model = models.Sequential([
    layers.Dense(64, activation='tanh'), # 레이어 / 노드수
    layers.Dense(128, activation='tanh'),
    layers.Dense(1, activation='sigmoid'), # 결과는 노드 한개. 결과값을 0~1사이로 압축해주는 sigmoid 사용.
]) # 모델 / 원하는 결과물를 도출하도록 모델을 설계하기

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) # binary_crossentropy는 확률문제에 적합한 손실함수
model.fit(train_x, train_y, epochs=1000) # numpy array로 변환

# 문제 예측
resume = pd.read_csv('download/data.csv')
resume.dropna()
resume = resume.values.tolist()
forcast_y = model.predict(resume)
# forcast_y = model.predict([[760, 3.0, 2], [400, 2.2, 1]])
print(forcast_y)