import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt

(trainX, trainY), (testX, testY) = tf.keras.datasets.fashion_mnist.load_data()

'''
print(trainX[0]) # 이미지 파일의 구조
print(trainX.shape) # 28x28 행렬이 6만개 있음
print(trainY) # 사진이 해당하는 라벨


plt.imshow(trainX[1])
# plt.show()
plt.gray()
plt.colorbar()
plt.savefig('preview.png')
'''

model = models.Sequential([
    layers.Dense(128, input_shape=(28,28), activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Flatten(),
    layers.Dense(10, activation="softmax"),
])
# 진위확률 예측문제 sigmoid / 마지막노드 1개
# 카테고리 예측문제 softmax / 마지막노드 카테고리 개수만큼
# sigmoid와 달리 softmax는 확률 총합이 1이 된다
# relu(rectified linear unit)은 음수는 0, 나머지는 원래값 그대로

model.summary()

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
# categorical_crossentropy는 리스트내 위치(원핫인코딩)
# sparse_categorical_crossentropy는 정수 카테고리
# 원핫인코딩으로 변환하려면 tf.keras.utils.to_categorical(pos,total) 사용

model.fit(trainX, trainY, epochs=5)