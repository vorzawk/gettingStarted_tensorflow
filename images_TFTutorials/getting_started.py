#import tensorflow as tf

# placeholder for inputs
x = tf.placeholder(tf.float32)
# linear_model is just a tensor but also represents the model used for learning.
W = tf.Variable(0.3, dtype=tf.float32)
b = tf.Variable(-0.3, dtype=tf.float32)
linear_model = W*x + b

# Initialize the variables defined earlier
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# linear_model can be evaluated for several values of x by feeding the dictionary of values.
# print(sess.run(linear_model, {x: [1.0, 2.0, 3.0, 4.0]}))

# The model has been created but we still need to check if it is any good. For this we need to define a loss function.

# Placeholder for training set outputs
y = tf.placeholder(tf.float32)

# Let's use the standard sum of squares error model since this is linear regression.
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

# Check to see if the perfect values for W and b produce a loss value of 0.
# Just assign the values to the variables
# fixW = tf.assign(W, [-1])
# fixb = tf.assign(b, [1])
# sess.run([fixb, fixW])
print("loss value(before training model): ", sess.run(loss, {x: [1.0, 2.0, 3.0, 4.0], y: [0, -1, -2, -3]}))
print("initial model parameters: ", sess.run([W, b]))

# Machine learning is not fun at all if you have to guess the model parameters, these
# must be found automatically.
# A gradient descent optimizer is used to train the model
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
for i in range(1000):
    sess.run(train, {x:[1, 2, 3, 4], y:[0, -1, -2, -3]})

print("loss value(after training model): ", sess.run(loss, {x: [1.0, 2.0, 3.0, 4.0], y: [0, -1, -2, -3]}))
print("trained model parameters: ", sess.run([W, b]))
