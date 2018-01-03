from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
sess = tf.InteractiveSession()

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
x_image = tf.reshape(x, [-1,28,28,1])

# convutional layer1 and pooling layer1
num_filter1 = 32

W_conv1 = weight_variable([5,5,1,num_filter1])
b_conv1 = bias_variable([num_filter1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# convutional layer2 and pooling layer2
num_filter2 = 64

W_conv2 = weight_variable([5,5,num_filter1,num_filter2])
b_conv2 = bias_variable([num_filter2])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# flatten the pool2 layer(7 * 7 * 64 = 3136) to one dimention vector
num_fc1_unit = 1024
num_all_pool = 28 / 2 / 2

W_fc1 = weight_variable([num_all_pool*num_all_pool*num_filter2, num_fc1_unit])
b_fc1 = bias_variable([num_fc1_unit])
h_pool2_flat = tf.reshape(h_pool2, [-1, num_all_pool*num_all_pool*num_filter2])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# add a dropout layer to avoid overfit
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 1024unit --> 10unit -->softmax
W_fc2 = weight_variable([num_fc1_unit, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# cross entropy implement by hand
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accurcy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.global_variables_initializer().run()
for i in range(500):
	batch = mnist.train.next_batch(50)
	if i%100 == 0:
		train_accurcy = sess.run(accurcy, feed_dict={x:batch[0], y_:batch[1], keep_prob:1.0})
		print "step %d, train_accurcy %g" % (i, train_accurcy)
	sess.run(train_step, feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})

print "test accurcy: %g" % sess.run(accurcy, feed_dict={x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0})
















