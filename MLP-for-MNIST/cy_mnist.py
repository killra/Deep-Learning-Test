import math
import tensorflow as tf

NUM_CLASSES = 10
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

def inference(images, hidden1_units, hidden2_units, keep_prob):
	with tf.name_scope('hidden1'):
		weights = tf.Variable(tf.truncated_normal([IMAGE_PIXELS, hidden1_units],
									stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),
							name='weights')
		#tf.add_to_collection("regular_loss", tf.contrib.layers.l1_regularizer(0.1)(weights))
		biases = tf.Variable(tf.zeros([hidden1_units]), name='biases')
		hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
		hidden1 = tf.nn.dropout(hidden1, keep_prob)
		#hidden1 = tf.sigmoid(tf.matmul(images, weights) + biases)
		#hidden1 = tf.tanh(tf.matmul(images, weights) + biases)
		#hidden1 = tf.nn.softplus(tf.matmul(images, weights) + biases)
		#hidden1 = tf.nn.elu(tf.matmul(images, weights) + biases)
		#hidden1 = tf.nn.leaky_relu(tf.matmul(images, weights) + biases)


	with tf.name_scope('hidden2'):
		weights = tf.Variable(tf.truncated_normal([hidden1_units, hidden2_units],
									stddev=1.0 / math.sqrt(float(hidden1_units))),
							name='weights')
		#tf.add_to_collection("regular_loss",tf.contrib.layers.l1_regularizer(0.1)(weights))
		biases = tf.Variable(tf.zeros([hidden2_units]), name='biases')
		hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
		hidden2 = tf.nn.dropout(hidden2, keep_prob)
		#hidden2 = tf.sigmoid(tf.matmul(hidden1, weights) + biases)
		#hidden2 = tf.tanh(tf.matmul(hidden1, weights) + biases)
		#hidden2 = tf.nn.softplus(tf.matmul(hidden1, weights) + biases)
		#hidden2 = tf.nn.elu(tf.matmul(hidden1, weights) + biases)
		#hidden2 = tf.nn.leaky_relu(tf.matmul(hidden1, weights) + biases)



	with tf.name_scope('softmax_linear'):
		weights = tf.Variable(tf.truncated_normal([hidden2_units, NUM_CLASSES],
									stddev=1.0 / math.sqrt(float(hidden2_units))),
							name='weights')
		biases = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases')
		logits = tf.nn.relu(tf.matmul(hidden2, weights) + biases)
		#logits = tf.sigmoid(tf.matmul(hidden2, weights) + biases)
		#logits = tf.tanh(tf.matmul(hidden2, weights) + biases)
		#logits = tf.nn.softplus(tf.matmul(hidden2, weights) + biases)
		#logits = tf.nn.elu(tf.matmul(hidden2, weights) + biases)
		#logits = tf.nn.leaky_relu(tf.matmul(hidden2, weights) + biases)



	return logits


def loss(logits, labels):
	labels = tf.to_int64(labels)
	function_loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
	#sum_loss = function_loss + regularLoss()

	return function_loss

def training(loss, learning_rate):
	tf.summary.scalar('loss', loss)
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	#optimizer = tf.train.AdamOptimizer()
	#optimizer = tf.train.AdagradOptimizer(learning_rate)
	#optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
	#optimizer = tf.train.RMSPropOptimizer(learning_rate)

	

	global_step = tf.Variable(0, name='global_step', trainable=False)
	train_op = optimizer.minimize(loss, global_step=global_step)
	return train_op

def evaluation(logits, labels):
	correct = tf.nn.in_top_k(logits, labels, 1)
	return tf.reduce_sum(tf.cast(correct, tf.int32))

def regularLoss():
	regular_loss = tf.add_n(tf.get_collection("regular_loss"))
	return regular_loss