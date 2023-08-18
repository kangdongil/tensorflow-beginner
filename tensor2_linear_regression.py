import tensorflow as tf

'''
heights = [170, 180, 175, 160]
shoe_sizes = [260, 270, 265, 255]

# shoe_size = a * height + b

height = heights[0]
shoe_size = shoe_sizes[0]

a = tf.Variable(0.1)
b = tf.Variable(0.2)

# a,b를 경사하강법으로 구하기
def loss_function():
    f_height = height * a + b
    return tf.square(shoe_size - f_height)

opt = tf.keras.optimizers.Adam(learning_rate=0.1)
for i in range(300):
    opt.minimize(loss_function, var_list=[a,b])
    print(a.numpy(),b.numpy(), height * a.numpy() + b.numpy())
'''

train_x = [1,2,3,4,5,6,7]
train_y = [3,5,7,9,11,13,15]

a = tf.Variable(0.1)
b = tf.Variable(0.2)

def loss_function(a, b):
    forcast_y = train_x * a + b
    return tf.keras.losses.mse(train_y, forcast_y)

opt = tf.keras.optimizers.Adam(learning_rate=0.01)
for i in range(3000):
    opt.minimize(lambda:loss_function(a, b), var_list=[a,b])
    print(f"a: {a.numpy()}, b: {b.numpy()}")