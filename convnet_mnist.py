import tensorflow as tf
# We need to define the weights and parameters for each of the NN layers, so make our lives easier by defining functions for doing this
def weight_variable(shape):
    # truncated_normal so that weights are not too far away from 0.0.
    initial = tf.truncated_normal( shape=shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    # small positive bias value so that we dont end with a lot of dead neurons using ReLU
    return tf.Variable(tf.constant(0.1, tf.float32, shape))

# We are using Vanilla Convnets with stride 1, so define functions to do this automatically
def conv2d(input, filter):
    return tf.nn.conv2d(input, filter, strides=[1,1,1,1], padding="SAME")

def max_pool(value):
    return tf.nn.max_pool(value, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

# Read input data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True);

# First Layer: ConvNet with 32 filters of size 5*5 and ReLU activation followed by pooling
Wconv1 = weight_variable([5, 5, 1, 32])
biasConv1 = bias_variable([32])

# Input data contains images flattened into 784*1 arrays, these must be reshaped into images
# Placeholder is needed since we want to run the training in batches
x = tf.placeholder(tf.float32, [None, 784])
x_image = tf.reshape(x, [-1,28,28,1])

conv1 = conv2d(x_image, Wconv1)
relu1 = tf.nn.relu(tf.add(conv1, biasConv1))
max_pool1 = max_pool(relu1)

# Second layer: ConvNet with 64 filters of size 5*5 and ReLU activation followed by pooling

Wconv2 = weight_variable([5, 5, 32, 64])
biasConv2 = bias_variable([64])

conv2 = conv2d(max_pool1, Wconv2)
relu2 = tf.nn.relu(tf.add(conv2, biasConv2))
max_pool2 = max_pool(relu2)

# Third layer: Fully Connected layer of 1024 neurons
# Image size goes from 28*28 -> 14*14 -> 7*7 as a result of the convolution layers. Now, each image has 7*7*64 pixel values which must be input to the FCC
max_pool2_flat = tf.reshape(max_pool2, [-1,7*7*64]) # Note that the order of the pixel values doesn't matter since all neurons are initialized the same way

Wfcc = weight_variable([7*7*64, 1024])
biasFcc = bias_variable([1024])
fcc = tf.nn.relu(tf.add(tf.matmul(max_pool2_flat, Wfcc), biasFcc))

# Fourth layer: Readout layer with 10 neurons

Wout = weight_variable([1024, 10])
biasOut = bias_variable([10])

# Do I need a relu here?
out = tf.nn.relu(tf.add(tf.matmul(fcc, Wout), biasOut))

# Set up placeholders for input labels
ye = tf.placeholder(tf.float32, [None, 10])
# Loss Function : Cross Entropy, Optimizer
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=ye, logits=out)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Training 
sess = tf.Session()
init_var = tf.global_variables_initializer()
sess.run(init_var)

for i in range(1000):
    print(i)
    batch_x, batch_ye = mnist.train.next_batch(100)
    sess.run(train_step, {x:batch_x, ye:batch_ye})

# Evaluation

# argmax returns the index of the largest value in the array which also doubles up as the classification
correct_prediction = tf.equal(tf.argmax(ye,1), tf.argmax(out,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, {x:mnist.test.images, ye:mnist.test.labels}))

