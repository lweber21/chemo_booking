import BookEnv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import csv
import Hyperparameters as param

#references
#https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0
#https://medium.freecodecamp.org/diving-deeper-into-reinforcement-learning-with-q-learning-c18d0db58efe



##Memory Hyperparameters
memory_size = 10

#Neural Network Parameters
n_inputs = param.DAYS + 1
n_hidden = 10
n_outputs = param.N_ACTIONS
learning_rate = param.ALPHA
eps = param.EPS

initializer = tf.contrib.layers.variance_scaling_initializer()


## Establish the neural network
X = tf.placeholder(shape=[None, param.STATES], dtype=tf.float32)
hidden = tf.layers.dense(X, n_hidden, activation=tf.nn.relu,  kernel_initializer=initializer)
Qout = tf.layers.dense(hidden, n_outputs, kernel_initializer=initializer)


predict = tf.argmax(Qout, 1)

#Calculate the loss function
nextQ = tf.placeholder(shape=[1, param.N_ACTIONS], dtype=tf.float32)
loss = tf.reduce_sum(tf.abs(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
updateModel = trainer.minimize(loss)



def update_explore(decay_step, eps_init=1, eps_final=0.01, decay_rate=0.0001):
    return eps_final + (eps_init - eps_final) * np.exp(-decay_rate * decay_step)


env = BookEnv.BookingEnv(days=param.DAYS, daily_avail=param.DAILY_CAP, demand_dist=param.DEMAND_DIST)

##Training the network
init = tf.global_variables_initializer()
eList = []
rList = []
dList = []
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    for i in range(param.N_EPISODES):
        env.reset()
        state = env.get_state()

        patient = env.get_patient()
        total_rewards = 0
        j = 0
        while j <= param.MAX_EPISODE_STEPS:
            j += 1

            a, allQ = sess.run([predict, Qout], feed_dict={X: state})

            if np.random.rand(1) < update_explore(j):
                a[0] = env.action_sample()
            next_state, next_patient, reward, done = env.step(a[0], patient)

            Q1 = sess.run(Qout, feed_dict={X: state})
            maxQ1 = np.max(Q1)
            targetQ = allQ
            targetQ[0, a[0]] = reward + param.GAMMA*maxQ1

            _,  = sess.run([updateModel], feed_dict={X: state, nextQ: targetQ})

            total_rewards += reward
            patient = next_patient
            state = next_state

            if done:
                e = 1./((i/50) + 10)
                break
        eList.append(i)

        rList.append(total_rewards)
        dList.append(env.total_demand)

        if(i + 1) % 100 == 0:
            print(f"Episode {i + 1}: total reward -> {total_rewards}")
            save_path = saver.save(sess, "./models/NN/model.ckpt")
            print("Model Saved at Episode: %i" % (i))

    plt.plot(eList, rList)
    plt.show()

    if param.WRITE_RESULTS:
        with open(param.training_fname, 'w', newline='') as f:
            thewriter = csv.writer(f)

            thewriter.writerow(["epoch", "total_reward", "total_demand"])
            for e in eList:
                thewriter.writerow([e + 1, rList[e], dList[e]])

    #lets put our agent to work
with tf.Session() as sess:
    saver.restore(sess, "./models/NN/model.ckpt")
    env.reset()
    state = env.get_state()
    patient = env.get_patient()

    total_rewards = 0
    j = 0
    while j <= param.MAX_EPISODE_STEPS:
        j += 1
        print(state)
        a, allQ = sess.run([predict,Qout], feed_dict={X: state})
        print(a[0])
        print(allQ)

        next_state, next_patient, reward, done = env.step(a[0], patient)

        total_rewards += reward
        patient = next_patient
        state = next_state

    print(f"total reward -> {total_rewards}")
    print(env)

    if param.WRITE_RESULTS:
        env.write_env(param.env_fname)