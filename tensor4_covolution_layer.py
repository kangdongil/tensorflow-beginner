import tensorflow as tf
keras = tf.keras
from keras import models, layers, datasets
import numpy as np
import matplotlib.pyplot as plt

(trainX, trainY), (testX, testY) = datasets.fashion_mnist.load_data()


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

'''
trainX = trainX / 255.0
trainY = trainY / 255.0
'''

trainX = trainX.reshape((*trainX.shape, 1)) # 흑백사진 1, 컬러사진 3 RGB
testX = testX.reshape((*testX.shape, 1))
# trainX = trainX.reshape((testX.shape[0], 28, 28, 1))
# print(trainX.shape, testX.shape)

model = models.Sequential([
    layers.Conv2D(32, (3,3), padding="same", activation="relu", input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    # .Conv2D(filter, kernel_size, padding)
    layers.Flatten(),
    # layers.Dense(128, input_shape=(28,28), activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax"),
])

# ndim=4, 4차원데이터를 요구하므로 input_shape과 데이터 차원을 수정한다

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
model.fit(trainX, trainY, validation_data=(testX,testY), epochs=5)

# 모든 eopch이 끝나고 평가할 때는 testX, testY를, epoch마다 평가를 하는 경우 valX, valY로 정의한다
'''
score = model.evaluate(testX, testY)
print(score)
'''