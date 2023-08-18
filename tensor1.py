import tensorflow as tf

'''
t = tf.constant(3)
print(t)
'''

'''
t1 = tf.constant([3,4,5])
t2 = tf.constant([6,7,8])
t3 = tf.constant([1,2,3], [4,5,6]) # 2x3 배열

# print(t1 + t2)
print(tf.add(t1, t2))
'''

'''
t1 = tf.zeros(5) # [0, 0, 0, 0, 0]
t2 = tf.zeros([2,2]) # [[0,0], [0,0]]
t3 = tf.zeros([2,2,3]) # [[[0,0,0], [0,0,0]], [[0,0,0], [0,0,0]]]

print(t1, t2, t3)
print(t3.shape)
'''

'''
t1 = tf.constant([5.0,4,3])
t2 = tf.constant([5,4,3], tf.float32)
print(t1, t2)

t3 = tf.constant([5,4,3])
print(t3)
t3 = tf.cast(t3, tf.float32)
print(t3)
'''

'''
w = tf.Variable(1.0)
w.assign(0.5)
print(w.numpy())
'''