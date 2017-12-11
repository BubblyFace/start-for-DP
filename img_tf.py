import tensorflow as tf 

#use session to get value of tf
 
g = tf.Graph()

g = tf.get_default_graph()

with g.as_default():
    x = tf.add(3, 5)

sess = tf.Session(graph = g)


# with tf.device('/gpu:0'):
#     a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0], name = 'a')
#     b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0], name = 'b')
#     c = tf.matmul(a,b)

# # with tf.device('/cpu:0'):
# #     c = tf.matmul(a,b)

# sess = tf.Session(config=tf.ConfigProto(log_devide_placement = True))

# sess.run(c)

sess.close()