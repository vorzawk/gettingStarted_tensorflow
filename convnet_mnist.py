import tensorflow as tf
# We need to define the weights and parameters for each of the NN layers, so make our lives easier by defining functions for doing this
def weight_variable(shape):
    # truncated_normal so that weights are not too far away from 0.0.
    initial = tf.truncated_normal( shape=shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    # small positive bias value so that we dont end with a lot of dead neurons using ReLU
    return tf.Variable(tf.constant(0.1, shape=shape))

# We are using Vanilla Convnets with stride 1, so define functions to do this automatically
def conv2d(input, filter):
    return tf.nn.conv2d(input, filter, strides=[1,1,1,1], padding="SAME")

def max_pool(value):
    return tf.nn.max_pool(value, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

# Read input data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True);

# First Layer: ConvNet with 32 filters of size 5*5 and ReLU activation followed by pooling
Wconv1 = weight_variable([5, 5, 1, 32]) # [filterDim1, filterDim2, filterDepth, NoOfFilters]
biasConv1 = bias_variable((32,))

# Input data contains images flattened into 784*1 arrays, these must be reshaped into images
# Placeholder is needed since we want to run the training in batches(so the dimension is only known at run time)
x = tf.placeholder(tf.float32, [None, 784])
x_image = tf.reshape(x, [-1,28,28,1])

# Set up placeholders for input labels
ye = tf.placeholder(tf.float32, [None, 10])

conv1 = conv2d(x_image, Wconv1)
relu1 = tf.nn.relu(conv1 + biasConv1)
max_pool1 = max_pool(relu1)

# Second layer: ConvNet with 64 filters of size 5*5 and ReLU activation followed by pooling
Wconv2 = weight_variable([5, 5, 32, 64])
biasConv2 = bias_variable((64,))

conv2 = conv2d(max_pool1, Wconv2)
relu2 = tf.nn.relu(conv2 + biasConv2)
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

# No need for ReLU here since this is the output layer, using ReLU simply makes things more complicated
out = tf.matmul(fcc, Wout) + biasOut

# Loss Function : Cross Entropy, Optimizer
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ye, logits=out))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# argmax returns the index of the largest value in the array which also doubles up as the classification
correct_prediction = tf.equal(tf.argmax(ye,1), tf.argmax(out,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy',accuracy)

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter("./log/train")
test_writer = tf.summary.FileWriter("./log/test")

# Training 
with tf.Session() as sess:
    init_var = tf.global_variables_initializer()
    sess.run(init_var)
    # 1000 steps of training
    for i in range(1000):
        batch_x, batch_ye = mnist.train.next_batch(100)
        sess.run(train_step, {x:batch_x, ye:batch_ye})
        # Print training and test set accuracies every 10 steps
        if i % 10 == 0:
            summary, acc = sess.run([merged, accuracy], {x:batch_x, ye:batch_ye})
            train_writer.add_summary(summary, i)
            summary, acc = sess.run([merged, accuracy], {x:mnist.test.images[0:500,:], ye:mnist.test.labels[0:500,:]})
            test_writer.add_summary(summary, i)

    # Evaluation
    # There are 10000 images in the test set, however running testing on all images at once causes the system to hang. Upon 
    # closer inspection, I realize that weird things start to happen for sizes 3500 and above.
    # For eg, pressing 'j' in vim causes forever to go to the next line, Ctrl+tab doesn't work in chrome and so on. 
    # Things work fine till dim=3000, but the time taken really shoots up after this. I would say that this is where swapping
    # to disk begins and the weird stuff happening is simply due to the required data and code getting evicted from the cache
    # to fill up the matrix values.
    # So, rather than running it all at a time, I will take the values 2000 at a time, compute the accuracies and average out
    # all the values. This would be much faster since the dimention isn't so small so as to not make use of the dense matrix
    # operations nor is it so big that it doesn't fit in memory!
    # The "weirdness" is still there coz such a huge computation does mess up the cache, at least all the training samples 
    # are evaluated in a resonable time now!

    aggr_accuracy = 0
    lim = 2000
    # Only 2000 iterations coz 10000 takes way too much time!
    for i in range(0,2000,lim):
#        aggr_accuracy += sess.run(accuracy, {x:mnist.test.images[i:i+lim-1,:], ye:mnist.test.labels[i:i+lim-1,:]})
        print(aggr_accuracy/5)

