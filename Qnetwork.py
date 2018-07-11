import tensorflow as tf
import numpy as np

class Qnetwork:
    #create a class to easily create multiple qnetworks
    def __init__(self, state_dimension, num_actions, hidden=[], learning_rate=1e-7):
        self.state_dimension = state_dimension
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.num_hidden_units = hidden
        self.layer = []
        self.initializer = tf.contrib.layers.variance_scaling_initializer()
        self.input = tf.placeholder(shape=[None, self.state_dimension], dtype=tf.float32)
        for i, n in enumerate(self.num_hidden_units):
            if i == 0:
                self.layer.append(tf.layers.dense(self.input, self.num_hidden_units[i],
                                                activation=tf.nn.relu, kernel_initializer=self.initializer))
            else:
                self.layer.append(tf.layers.dense(self.layer[i-1], self.num_hidden_units[i],
                                                activation=tf.nn.relu, kernel_initializer=self.initializer))

        self.Qout = tf.layers.dense(self.layer[-1], self.num_actions, kernel_initializer=self.initializer)
        self.predict = tf.argmax(self.Qout, 1)

        # Calculate the loss function
        self.targetQ = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None,1], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, depth=self.num_actions, dtype=tf.float32)
        self.Q = tf.reduce_sum(tf.multiply(self.actions_onehot, self.Qout))

        self.loss = tf.reduce_mean(tf.square(self.targetQ - self.Q))
        self.trainer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        self.updateModel = self.trainer.minimize(self.loss)


#myNet = Qnetwork(10, 4, hidden=[3,4])