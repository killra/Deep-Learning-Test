from datetime import datetime
import math
import time
import tensorflow as tf

batch_size = 32
num_batches = 5

def print_activations(t):
	print t.op.name, " ", t.get_shape().as_list()

def variable_with_weight_loss(shape, stddev, wl):
	var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
	if wl is not None:
		weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name='weight_loss')
		tf.add_to_collection('losses', weight_loss)
	return var

def loss(logits, labels):
	#labels = tf.cast(labels, tf.int64)
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
		logits=logits, labels=labels, name='cross_entropy_per_example')
	cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
	tf.add_to_collection('losses', cross_entropy_mean)
	return tf.add_n(tf.get_collection('losses'), name='total_loss')

def inference(images):
	parameters = []

	with tf.name_scope('conv1') as scope:
		kernel = tf.Variable(tf.truncated_normal([11,11,3,96], dtype=tf.float32, stddev=1e-1), name='weights')
		conv = tf.nn.conv2d(images, kernel, [1,4,4,1], padding='VALID')
		biases = tf.Variable(tf.constant(0.0, shape=[96], dtype=tf.float32), trainable=True, name='biases')
		bias = tf.nn.bias_add(conv, biases)
		conv1 = tf.nn.relu(bias, name=scope)
		print_activations(conv1)
		parameters += [kernel, biases]

		lrn1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001/9, beta=0.75, name='lrn1')
		pool1 = tf.nn.max_pool(lrn1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID', name='pool1')
		print_activations(pool1)

	with tf.name_scope('conv2') as scope:
		kernel = tf.Variable(tf.truncated_normal([5,5,96,256], dtype=tf.float32, stddev=1e-1), name='weights')
		conv = tf.nn.conv2d(pool1, kernel, [1,1,1,1], padding='SAME')
		biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
		bias = tf.nn.bias_add(conv, biases)
		conv2 = tf.nn.relu(bias, name=scope)
		print_activations(conv2)
		parameters += [kernel, biases]

		lrn2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001/9, beta=0.75, name='lrn2')
		pool2 = tf.nn.max_pool(lrn2, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID', name='pool2')
		print_activations(pool2)

	with tf.name_scope('conv3') as scope:
		kernel = tf.Variable(tf.truncated_normal([3,3,256,384], dtype=tf.float32, stddev=1e-1), name='weights')
		conv = tf.nn.conv2d(pool2, kernel, [1,1,1,1], padding='SAME')
		biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32), trainable=True, name='biases')
		bias = tf.nn.bias_add(conv, biases)
		conv3 = tf.nn.relu(bias, name=scope)
		print_activations(conv3)
		parameters += [kernel, biases]

	with tf.name_scope('conv4') as scope:
		kernel = tf.Variable(tf.truncated_normal([3,3,384,384], dtype=tf.float32, stddev=1e-1), name='weights')
		conv = tf.nn.conv2d(conv3, kernel, [1,1,1,1], padding='SAME')
		biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32), trainable=True, name='biases')
		bias = tf.nn.bias_add(conv, biases)
		conv4 = tf.nn.relu(bias, name=scope)
		print_activations(conv4)
		parameters += [kernel, biases]

	with tf.name_scope('conv5') as scope:
		kernel = tf.Variable(tf.truncated_normal([3,3,384,256], dtype=tf.float32, stddev=1e-1), name='weights')
		conv = tf.nn.conv2d(conv4, kernel, [1,1,1,1], padding='SAME')
		biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
		bias = tf.nn.bias_add(conv, biases)
		conv5 = tf.nn.relu(bias, name=scope)
		print_activations(conv5)
		parameters += [kernel, biases]	

		pool5 = tf.nn.max_pool(conv5, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID', name='pool5')
		print_activations(pool5)

	with tf.name_scope('dense1') as scope:
		reshape = tf.reshape(pool5, [-1, 6*6*256])
		weight = variable_with_weight_loss(shape=[6*6*256, 4096], stddev=0.01, wl=0.0)
		bias = tf.Variable(tf.constant(0.1, shape=[4096]))
		dense1 = tf.nn.relu(tf.matmul(reshape, weight) + bias)
		print_activations(dense1)

	with tf.name_scope('dense2') as scope:
		weight = variable_with_weight_loss(shape=[4096, 4096], stddev=0.01, wl=0.0)
		bias = tf.Variable(tf.constant(0.1, shape=[4096]))
		dense2 = tf.nn.relu(tf.matmul(dense1, weight) + bias)
		print_activations(dense2)

	with tf.name_scope('dense3') as scope:
		weight = variable_with_weight_loss(shape=[4096, 1000], stddev=0.01, wl=0.0)
		bias = tf.Variable(tf.constant(0.1, shape=[1000]))
		logits = tf.add(tf.matmul(dense2, weight), bias)
		print_activations(logits)


	return logits, pool5, parameters	

def time_tensorflow_run(session, target, info_string):
	num_steps_burn_in = 10
	total_duration = 0.0
	total_duration_squared = 0.0

	for i in range(num_batches+num_steps_burn_in):
		start_time = time.time()
		_ = session.run(target)
		duration = time.time() - start_time
		if i >= num_steps_burn_in:
			if not(i%10):
				print '%s: step %d, duration = %.3f' % (datetime.now(), i - num_steps_burn_in, duration)
			total_duration += duration
			total_duration_squared += duration * duration

	mn = total_duration / num_batches
	vr = total_duration_squared / num_batches - mn * mn
	sd = math.sqrt(vr)
	print '%s: %s across %d steps, %.3f +/- %.3f sec / batch' % (datetime.now, info_string, num_batches, mn, sd)

def run_benchmark():
	with tf.Graph().as_default():
		image_size = 224
		images = tf.Variable(tf.random_normal([batch_size, 227, 227, 3], dtype=tf.float32, stddev=1e-1))
		labels = tf.Variable(tf.random_uniform([batch_size], maxval=10, dtype=tf.int64))

		logits, pool5, parameters = inference(images)
		loss_op = loss(logits, labels)
		train_op = tf.train.AdamOptimizer(1e-3).minimize(loss_op)
		top_k_op = tf.nn.in_top_k(logits, labels, 1)

		init = tf.global_variables_initializer()
		sess = tf.Session()
		sess.run(init)

		time_tensorflow_run(sess, pool5, "Forward")
		time_tensorflow_run(sess, loss_op, "Train")


def main(_):
  run_benchmark()

if __name__ == '__main__':
	tf.app.run(main=main)
