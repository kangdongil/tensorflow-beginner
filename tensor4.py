import tensorflow as tf
import matplotlib.pyplot as plt

(trainX, trainY), (testX, testY) = tf.keras.datasets.fashion_mnist.load_data()


print(trainX[0]) # 이미지 파일의 구조
print(trainX.shape) # 28x28 행렬이 6만개 있음
print(trainY) # 사진이 해당하는 라벨


plt.imshow(trainX[1])
# plt.show()
plt.gray()
plt.colorbar()
plt.savefig('preview.png')